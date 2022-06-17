import get_data
import preprocess
import base_models
import learner

# To be used sparingly, mostly to make sure a far-reaching change hasn't messed something up
def run_entire_pipeline():
    get_data.main()
    preprocess.main()
    base_models.main()
    learner.main()

def main():
    run_entire_pipeline()

if __name__ == "__main__":
    main()
