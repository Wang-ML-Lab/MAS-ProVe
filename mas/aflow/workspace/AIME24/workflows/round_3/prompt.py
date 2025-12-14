ANALYSIS_PROMPT = """You are a careful mathematical analyst. Given the problem below, do the following in a structured way:

1) Restate the problem in one sentence.
2) List clear high-level approaches (e.g., reduce to counting fixed points, use Hensel lifting, enumerate small cases, symmetry and orbit-counting).
3) For each promising approach, provide a short concrete plan of steps to get an exact integer answer (no vague language). If a numerical enumeration or modular lifting is needed, explicitly say so and specify the ranges and formulas to compute.
4) If you can deduce the final integer answer purely by reasoning, give the exact integer and a concise justification (tag the final answer as ANSWER: <number> on its own line). If you cannot compute the final integer without calculation, state which computations should be delegated to a rigorous computation step (Programmer operator).
5) Keep the output concise and structured with labeled sections: RESTATE, APPROACHES, PLANS, and either ANSWER or COMPUTE.

Now analyze this problem and produce the structured response. Do not include placeholders or external tool calls; just produce the analysis text."""