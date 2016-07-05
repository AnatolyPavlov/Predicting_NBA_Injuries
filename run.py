import get_data
import preprocess
import model

if __name__ == '__main__':
    get_data.get_data()
    preprocess.preprocess()
    model.main()