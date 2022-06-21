import pipeline.get_data
import pipeline.preprocess
import pipeline.base_models
import pipeline.learner

# To be used sparingly, mostly to make sure a far-reaching change hasn't messed something up
def run_entire_pipeline():
    pipeline.get_data.main()
    pipeline.preprocess.main()
    pipeline.base_models.main()
    pipeline.learner.main()

def main():
    run_entire_pipeline()

if __name__ == "__main__":
    main()
