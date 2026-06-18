"""Entry point: run one default peloton race and print the result.

    uv run python main.py

For parameter sweeps / sensitivity analysis use ``python -m peloton.sweep``;
for across-race learning use ``peloton.evolution.run_generations``.
"""

from peloton.model import PelotonModel


def main(max_steps: int = 200) -> None:
    model = PelotonModel()
    for _ in range(max_steps):
        if not model.running:
            break
        model.step()

    print(f"Race over after {model.steps} steps; {model.n_finished} riders finished.")
    for rank, (unique_id, step) in enumerate(model.finish_order, start=1):
        print(f"  {rank:>2}. rider {unique_id} (step {step})")


if __name__ == "__main__":
    main()
