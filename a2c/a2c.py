import numpy
import torch
from a2c.base import BaseAlgo


class A2CAlgo(BaseAlgo):
    """The Advantage Actor-Critic algorithm."""

    def __init__(
        self,
        envs,
        acmodel,
        device=None,
        num_frames_per_proc=None,
        discount=0.99,
        lr=0.01,
        gae_lambda=0.95,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=0.5,
        recurrence=4,
        rmsprop_alpha=0.99,
        rmsprop_eps=1e-8,
        preprocess_obss=None,
        reshape_reward=None,
    ):
        num_frames_per_proc = num_frames_per_proc or 8

        super().__init__(
            envs,
            acmodel,
            device,
            num_frames_per_proc,
            discount,
            lr,
            gae_lambda,
            entropy_coef,
            value_loss_coef,
            max_grad_norm,
            recurrence,
            preprocess_obss,
            reshape_reward,
        )

        self.optimizer = torch.optim.RMSprop(
            self.acmodel.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps
        )

    def update_parameters(self, exps):
        # P is the number of processes
        # T is the number of frames per process
        # D is the dimensionality of the observation space
        # exps.obs: (P * T) x D
        # exps.memory: (P * T) x D
        # exps.mask: (P * T) x 1
        # exps.action: (P * T)
        # exps.value: (P * T)
        # exps.reward: (P * T)
        # exps.advantage: (P * T)
        # exps.returnn: (P * T)
        # exps.log_prob: (P * T)

        # Ok currently,
        # P=16
        # T=8
        # total_frames=128
        # Recurrence=4
        # Dimensions of memory=128

        # Compute starting indexes
        # Total frames:
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18...
        # Frames 0 to 7 are the frames for process 0
        # Frames 8 to 15 are the frames for process 1, and so on.
        # Since recurrence is 4, we use these as the starting indices:
        # 0, 4, 8, 12, 16, etc.

        # Since 128 / 4 = 32,
        # inds has 32 indices
        inds = self._get_starting_indexes()

        # Initialize update values

        update_entropy = 0
        update_value = 0
        update_policy_loss = 0
        update_value_loss = 0
        update_loss = 0

        # Initialize memory

        if self.acmodel.recurrent:
            # This is the starting memory of all the inds
            memory = exps.memory[inds]

        for i in range(self.recurrence):
            # Creates a sub-batch of experience
            # We can think of sb as containing all the information in exps for a single
            # time step for all 4-step recurrences we are currently processing.
            sb = exps[inds + i]

            # Compute loss
            if self.acmodel.recurrent:
                dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
            else:
                dist, value, _ = self.acmodel(sb.obs)

            entropy = dist.entropy().mean()

            policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

            value_loss = (value - sb.returnn).pow(2).mean()

            loss = (
                policy_loss
                - self.entropy_coef * entropy
                + self.value_loss_coef * value_loss
            )

            # Update batch values

            update_entropy += entropy.item()
            update_value += value.mean().item()
            update_policy_loss += policy_loss.item()
            update_value_loss += value_loss.item()
            update_loss += loss

        # Update update values

        update_entropy /= self.recurrence
        update_value /= self.recurrence
        update_policy_loss /= self.recurrence
        update_value_loss /= self.recurrence
        update_loss /= self.recurrence

        # Update actor-critic

        self.optimizer.zero_grad()
        update_loss.backward()
        update_grad_norm = (
            sum(p.grad.data.norm(2) ** 2 for p in self.acmodel.parameters()) ** 0.5
        )
        torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Log some values

        logs = {
            "entropy": update_entropy,
            "value": update_value,
            "policy_loss": update_policy_loss,
            "value_loss": update_value_loss,
            "grad_norm": update_grad_norm,
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes
