import numpy as np
import wine_quality_data
import human_activity_data
from problem3 import SimpleClassifier


def main():
    cf = SimpleClassifier()
    wine_quality_data.run_wine_inferences(cf)
    #human_activity_data.run_activity_inferences(cf)



if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    main()
