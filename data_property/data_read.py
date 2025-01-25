import pickle
import os

data_path = './data_list'

test_file_list = os.path.join(data_path,'val_list.pkl')
with open("valfile.txt", "w") as txt:
    if os.path.exists(test_file_list):
        with open(test_file_list, "rb") as f:
            test_datalist = pickle.load(f)
            new_list = []
            # print(test_datalist, len(test_datalist))
            count = 0
            for path in test_datalist:
                path = './'+path[path.index('/')+10:]
                txt.write(path + "\n")
                count +=1
            print(f'count:{count}')


            # print(new_list, len(new_list))