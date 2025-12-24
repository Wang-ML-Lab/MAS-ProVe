from mas_proceval.mas.mas_base import MASBase

class MASDyLAN(MASBase):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Initialize any DyLAN-specific state here
        # e.g., self.config, self.roles, self.rounds, etc.

    async def run(self):
        """
        Main entry point for DyLAN process evaluation.
        Override this method to implement the multi-round, multi-agent workflow.
        Use self.trajectory to maintain context across rounds/agents.
        """
        raise NotImplementedError("Implement DyLAN's main workflow here.")

    async def generate_agent_response(self, agent_idx, context, *args, **kwargs):
        """
        Generate a response for a single agent in a given round.
        Decorate this method for agent-level process evaluation if needed.
        """
        raise NotImplementedError("Implement agent response generation here.")

    async def aggregate_round(self, round_idx, agent_responses, *args, **kwargs):
        """
        Aggregate or judge all agent responses for a round.
        Optionally use process evaluation to select the best or combine outputs.
        """
        raise NotImplementedError("Implement round aggregation/judging here.")

    # Add more abstract methods as needed for your DyLAN workflow
