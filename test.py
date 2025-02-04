import pickle

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except EOFError:
    print("Error: The file is empty or corrupt. Please check the file.")