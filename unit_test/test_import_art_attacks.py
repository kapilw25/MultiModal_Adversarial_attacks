#!/usr/bin/env python3

# Import the attacks from ART library
from art.attacks.evasion import SquareAttack
from art.attacks.evasion import HopSkipJump

# Print confirmation that imports were successful
print("Successfully imported SquareAttack and HopSkipJump from ART library")

# # Example of how to use these attacks (for reference)
# def example_usage():
#     print("\nExample usage (not executed):")
#     print("# Create a classifier")
#     print("classifier = create_classifier()")
#     print("\n# Create Square Attack")
#     print("square_attack = SquareAttack(")
#     print("    estimator=classifier,")
#     print("    norm=np.inf,")
#     print("    max_iter=100,")
#     print("    eps=0.3,")
#     print("    p_init=0.8,")
#     print("    nb_restarts=1,")
#     print("    batch_size=128,")
#     print("    verbose=True")
#     print(")")
#     print("\n# Create HopSkipJump Attack")
#     print("hop_skip_jump = HopSkipJump(")
#     print("    classifier=classifier,")
#     print("    batch_size=64,")
#     print("    targeted=False,")
#     print("    norm=2,")
#     print("    max_iter=50,")
#     print("    max_eval=10000,")
#     print("    init_eval=100,")
#     print("    init_size=100,")
#     print("    verbose=True")
#     print(")")

# if __name__ == "__main__":
#     example_usage()