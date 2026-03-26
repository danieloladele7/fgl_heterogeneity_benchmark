"""Run the full self-contained validation workflow."""

from validation.run_all_validation import synthetic_main, karate_main, invariants_main

if __name__ == "__main__":
    invariants_main()
    karate_main()
    synthetic_main()
