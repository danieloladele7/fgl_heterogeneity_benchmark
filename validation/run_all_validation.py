from validation.test_protocol_invariants import main as invariants_main
from validation.validate_karate import main as karate_main
from validation.validate_synthetic_domains import main as synthetic_main

if __name__ == "__main__":
    invariants_main()
    karate_main()
    synthetic_main()
    print("All validation stages completed successfully.")
