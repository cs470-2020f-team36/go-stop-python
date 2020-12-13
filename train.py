"""
train.py

Train the neural network.
Note that this project does not use GPU,
as the most time-consuming step in this project is MCTS,
which is difficult to accelerate using GPU.
"""

from __future__ import annotations
import json
import math
import pickle
import random
from typing import Dict, List, Optional, Tuple

import copy
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
import tqdm

from go_stop.models.agent import Agent
from go_stop.models.game import Game
from go_stop.models.action import (
    Action,
    NUM_ACTIONS,
    ALL_ACTIONS,
    get_action_index,
)
from go_stop.train.args import args
from go_stop.train.encoder import DIM_ENCODED_GAME, encode_game
from go_stop.train.match import match_agents, is_first_better
from go_stop.train.network import AlphaLoss, EncoderNet
from go_stop.train.reward import reward_wrt_player
from go_stop.train.sampler import sample_from_observation
from go_stop.utils.exp import mean_exp


torch.manual_seed(args.seed)


# `P` is a dict used to store the prior probability of nodes.
# A dict with observable states as keys and policies as values
# Dict[tuple of length args.ninp, Tensor of shape (args.nout,)]
P: Dict[Tuple[int, ...], Tensor] = {}


class UCTNode:
    """
    Implement UCTNode.

    Refer to https://web.stanford.edu/~surag/posts/alphazero.html
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        game: Game,
        parent: Optional[UCTNode] = None,
        action: Optional[Action] = None,
    ):
        self.game = game
        self.parent = parent
        # self.parent[self.action] == self
        # None if self is the root node
        self.action = action

        # Whether `self` is visited at least once or not
        self.is_expanded = False
        # The key is the index of the action
        self.children: Dict[int, UCTNode] = {}

        # Q(s, a); the key is the index of the action
        self._q: Dict[int, float] = {}
        # N(s, a); the key is the index of the action
        self._n: Dict[int, int] = {}

    def encoded_game(self) -> tuple:
        """Return the encoded game w.r.t. the current player."""
        encoded_game = encode_game(self.game, self.game.state.player)
        return tuple(encoded_game.int().tolist())

    def p(self, action: Action) -> float:
        """Return the prior probability distribution stored in `P`."""
        encoded_game = self.encoded_game()
        assert encoded_game in P
        return P[encoded_game][get_action_index(action)]

    def policy(self, tau: float = 1) -> Tensor:
        r"""
        Return the policy defined by the MCTS:
            \pi(a \mid s; \tau) \propto N(s, a)^{1 / \tau}
        """
        n_tensor = Tensor([self.n(action) for action in ALL_ACTIONS])
        policy = mean_exp(n_tensor, 1 / tau)
        return policy

    def policy_with_noise(self, tau: float = 1) -> Tensor:
        r"""
        Return the weighted average of the policy from MCTS
        and the Dirichlet noise:
            (1 - \epsilon) \pi(\bullet \mid s; \tau) + \epsilon E,
            E \sim Dirichlet(\alpha, \dots, \alpha)
        """
        policy = self.policy(tau)
        dirichlet_noise = torch.from_numpy(
            np.random.dirichlet(np.zeros([NUM_ACTIONS], dtype=np.float32) + args.alpha)
        )
        policy = (1 - args.epsilon) * policy + args.epsilon * dirichlet_noise
        mask = (
            Tensor(
                [
                    1 if ALL_ACTIONS[i] in self.game.actions() else 0
                    for i in range(NUM_ACTIONS)
                ],
            )
            == 0
        )
        policy = policy.masked_fill(mask, 0)
        policy = policy / torch.sum(policy)

        return policy

    def q(self, action: Action) -> float:
        """Return Q(s, a)."""
        if get_action_index(action) in self._q:
            return self._q[get_action_index(action)]
        return 0

    def set_q(self, action: Action, q_value: float):
        """Set Q(s, a)."""
        self._q[get_action_index(action)] = q_value

    def n(self, action: Action) -> int:
        """Return N(s, a)."""
        if get_action_index(action) in self._n:
            return self._n[get_action_index(action)]
        return 0

    def set_n(self, action: Action, n_value: int):
        """Set N(s, a)."""
        self._n[get_action_index(action)] = n_value

    def u(self, action: Action) -> float:
        r"""
        Return U(s, a), which is defined by
        U(s, a) = Q(s, a) + c_{puct} ⋅ P(s, a) ⋅ \frac{\sqrt{\sum_n N(s, b)}}{1 + N(s, a)}.
        """
        r = self.q(action) + args.c_puct * self.p(action) * math.sqrt(
            sum(self._n.values())
        ) / (1 + self.n(action))
        if r.isnan():
            print(self.q(action))
            print(self.p(action))
            print(sum(self._n.values()))
            print(self.n(action))
            raise Exception("U calculation")
        return r


def search(node: UCTNode, net: EncoderNet, action: Optional[Action] = None) -> float:
    """
    Search through the tree,
    and return the expected reward for the current player of the parent node.

    node: the current node
    net: the neural network
    action: this parameter is given only when the neural network predicts an action
            while the corresponding child to the action is not visited yet.
    """

    # terminal state
    if node.game.state.ended:
        #  |   (+)   P0    (-)   |
        #  |  ---->   O   <----  |
        #  |  |     /   \     |  |
        #  |  --- O       O ---  |
        #  |     P0       P1     |

        return reward_wrt_player(node.game, node.parent.game.state.player)

    # Since the node is not the terminal state,
    # `node.game.actions()` must not be empty
    assert node.game.actions() != []

    # During a normal search process
    if action is None:
        # find the action maximizing U(s, a)
        max_u_value = -float("inf")

        for action_ in node.game.actions():
            u_value = node.u(action_)
            if u_value > max_u_value:
                max_u_value = u_value
                action = action_

    if get_action_index(action) in node.children:
        # If the child corresponding to the chosen action has been visited,
        # get the child node
        next_node = node.children[get_action_index(action)]
        value = search(next_node, net)

    else:
        # Elsewise:
        # Create the children and set the game state
        next_game = copy.deepcopy(node.game)
        next_game.play(action)

        # Store the child node in `node.children`
        next_node = UCTNode(next_game, node, action)
        node.children[get_action_index(action)] = next_node

        if next_game.state.ended:
            # If the game of the child node is ended,
            # get the reward from its final score
            value = reward_wrt_player(node.game, node.game.state.player)

        else:
            # Elsewise,
            encoded_game = Tensor(next_node.encoded_game()).unsqueeze(0)
            net.eval()
            with torch.no_grad():
                policy, value = net(encoded_game)

                # get the reward from the neural network,
                value = float(value)

                # and store the predicted prior into `P`
                encoded_game_tuple = tuple(encoded_game.squeeze().int().tolist())
                P[encoded_game_tuple] = policy.squeeze()
                if next_game.state.player != node.game.state.player:
                    value *= -1

    # Update the N(s, a) and Q(s, a)
    n_value = node.n(action) + 1
    q_value = (node.n(action) * node.q(action) + value) / n_value
    node.set_q(action, q_value)
    node.set_n(action, n_value)

    # Note `value` is the reward for the current node.
    # Multiply -1 to change the player if necessary, and return it.
    return value * (
        1
        if node.parent is None
        or node.game.state.player == node.parent.game.state.player
        else -1
    )


def create_root_node(game: Game, net: EncoderNet):
    """Create the root node and store the prior into `P`."""
    node = UCTNode(game)
    encoded_game = Tensor(node.encoded_game()).float().unsqueeze(0)
    net.eval()
    with torch.no_grad():
        policy, _ = net(encoded_game)
        P[node.encoded_game()] = policy.squeeze()

    return node


def execute_episode(net: EncoderNet) -> Tensor:
    """
    Execute the MCTS for a game (and its siblings)
    until the game of the root node is terminated.
    """

    training_data = torch.zeros((0, DIM_ENCODED_GAME + NUM_ACTIONS + 2))
    root_node = create_root_node(Game(), net)

    while True:
        # Generate a training datum
        num_of_cards_in_hand = len(root_node.game.board.hands[0]) + len(
            root_node.game.board.hands[1]
        )

        # Take a sample of size `args.num_similar_games(num_of_cards_in_hand)`
        # consisting of games with the same observable information.
        similar_games = sample_from_observation(
            root_node.game,
            root_node.game.state.player,
            sample_size=args.num_similar_games(num_of_cards_in_hand),
        )

        # Create the corresponding nodes.
        # In this case, we do not need to append the game state to `P`
        # since the observed information will be the same with `root_node.game`.
        similar_nodes = [root_node] + [
            create_root_node(game, net) for game in similar_games
        ]

        # We will take the average over the priors and the expected values
        # of those sibling games.
        average = torch.zeros((0, DIM_ENCODED_GAME + NUM_ACTIONS + 2))

        for node in tqdm.tqdm(similar_nodes, desc="similar_nodes", leave=False):
            # Do the MCTS simulation with the neural network
            for _ in tqdm.tqdm(
                range(args.mcts_search_per_simul),
                desc="mcts_search",
                leave=False,
            ):
                search(node, net)

            tau = (
                1
                if num_of_cards_in_hand < args.tau_threshold
                else args.infinitesimal_tau
            )
            action_index = np.random.choice(
                range(NUM_ACTIONS),
                p=node.policy(tau=args.infinitesimal_tau).clone().numpy(),
            )
            action = ALL_ACTIONS[action_index]
            training_datum = torch.cat(
                (
                    Tensor(node.encoded_game()),
                    node.policy(tau),
                    Tensor([node.q(action)]),
                    Tensor([node.game.state.player]),
                )
            ).unsqueeze(0)

            # It should not contain any NaN, which will cause a failure of training process.
            if training_datum.isnan().any():
                raise Exception("training_datum contains an NaN")

            # Append the training datum
            average = torch.cat((average, training_datum))

        # Average out all the training data gathered
        average = average.mean(axis=0).unsqueeze(0)

        # And append to the final data
        training_data = torch.cat((training_data, average))

        # Choose the best action
        action_index = np.random.choice(
            NUM_ACTIONS, p=list(root_node.policy_with_noise())
        )
        # The following will ensure that `action_index` have been visited at least once
        while action_index not in root_node.children:
            search(root_node, net, ALL_ACTIONS[action_index])

        # Take the best action we chose
        root_node = root_node.children[action_index]
        if root_node.game.state.ended:
            # If the game of the root node terminated, exit the loop
            break

    # Fill the rewards into the training data
    for datum_count in range(training_data.size(0)):
        player = int(training_data[datum_count, -1].item())
        avg_score = training_data[datum_count, -2].item()
        training_data[datum_count, -2] = (
            args.mcts_reward_weight * reward_wrt_player(root_node.game, player=player)
            + (1 - args.mcts_reward_weight) * avg_score
        )

    return training_data[:, :-1]


def evolve(net, evolution_count: int) -> Tuple[EncoderNet, float]:
    """
    Make the neural network evolve
    by executing the MCTS and training with the data provided by it.
    """

    net.train()

    # Define the optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.learning_rate * (args.lr_decrease_rate ** evolution_count),
    )

    try:
        with open(
            args.root_dir
            / f"training_data_{args.num_hidden_layers}_hidden_layers.pickle",
            "rb",
        ) as training_data_pickle:
            examples = pickle.load(training_data_pickle)
    except:
        print("NO SAVED EXAMPLES")
        examples = torch.zeros((0, DIM_ENCODED_GAME + NUM_ACTIONS + 1))

    for _ in tqdm.tqdm(
        range(args.num_episodes_per_evolvution), desc="episodes"
    ):
        # Gather the training data
        training_data = execute_episode(net)
        examples = torch.cat((examples, training_data))

        with open(
            args.root_dir
            / f"training_data_{args.num_hidden_layers}_hidden_layers.pickle",
            "wb",
        ) as training_data_pickle:
            pickle.dump(examples, training_data_pickle)

        trimmed_examples = examples[-args.replay_buffer_size(evolution_count) :]

        # Permute it randomly
        trimmed_examples = trimmed_examples[torch.randperm(trimmed_examples.size(0))]

        # Split it into minibatches
        minibatches = torch.split(trimmed_examples, args.batch_size)

        total_policy_loss = 0
        total_value_loss = 0

        for batch in minibatches:
            # If the batch size is (unfortunately) 1, ignore it.
            if batch.size(0) == 1:
                break

            # Tensor of shape (B, args.ninp)
            states = batch[:, :DIM_ENCODED_GAME]

            # Target data from the MCTS.
            # Tuple of tensors of shape (B, args.nout) and shape (B,), resp.
            y_target = (
                batch[:, DIM_ENCODED_GAME:-1],
                batch[:, -1:].squeeze(),
            )
            # Predicted data from the NN.
            y_pred = net(states)

            # Do the backpropagation.
            criterion = AlphaLoss()
            loss, policy_loss, value_loss = criterion(y_pred, y_target)
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            if loss.isnan():
                print(y_target, y_pred)
                raise Exception

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_policy_loss /= len(minibatches)
        total_value_loss /= len(minibatches)
        total_loss = total_policy_loss + total_value_loss

        print(f"Average loss: {total_loss}")

    return net


def main():
    """Main training process."""

    # pylint: disable=global-statement
    global P

    net = EncoderNet()

    # Load state dict if exists
    ckpt_path = args.root_dir / f"best_{args.num_hidden_layers}_hidden_layers.pt"
    if ckpt_path.is_file():
        net.load_state_dict(torch.load(ckpt_path))

    # Load env
    env_path = args.root_dir / f"env_{args.num_hidden_layers}_hidden_layers.json"
    if env_path.is_file():
        with open(env_path, "r") as env_file:
            env = json.load(env_file)
    else:
        env = {"version": 0}
        with open(env_path, "w") as env_file:
            json.dump(env, env_file)

    best_net = copy.deepcopy(net)

    for evolution_count in tqdm.tqdm(range(args.max_evolution), desc="evolution"):
        if evolution_count >= env["version"]:
            # Initialize `P`
            P = {}

            print(f"[Evolution {evolution_count}]")

            # Update the network
            net = evolve(net, evolution_count)

            # Match the trained network with the previous one.
            prev_agent = Agent.from_net(best_net)
            agent = Agent.from_net(net)
            points = match_agents(agent, prev_agent)
            is_evolved = is_first_better(points)

            # Load state dict if exists
            torch.save(
                net.state_dict(),
                args.root_dir / f"now_{args.num_hidden_layers}_hidden_layers.pt",
            )

            if is_evolved:
                print("The new network won the previous best one.")
                torch.save(
                    net.state_dict(),
                    args.root_dir / f"best_{args.num_hidden_layers}_hidden_layers.pt",
                )
                best_net = net
            else:
                print("The previous best network won the new one.")
                net = best_net

            env["version"] += 1

            # Record the match with a random agent.
            points = match_agents(Agent.from_net(net), Agent.random())

            try:
                env["evaluations"] = env["evaluations"]
            except KeyError:
                env["evaluations"] = {"mcts vs random": {}}

            env["evaluations"]["mcts vs random"][str(evolution_count)] = {
                "non_defeat_rate": len([x for x in points if x >= 0]) / len(points),
                "win_rate": len([x for x in points if x > 0]) / len(points),
                "mean": np.mean(points),
                "std": np.std(points),
            }

            with open(env_path, "w") as env_file:
                json.dump(env, env_file)


if __name__ == "__main__":
    main()
