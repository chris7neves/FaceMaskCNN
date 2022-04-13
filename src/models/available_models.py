from models.fmcnn1 import get_fmcnn1
from models.fmcnn2 import get_fmcnn2

# Dict where key is the model name as a string
# and the value is a reference to the function used 
# assembly everything necessary for model usage
model_dict ={
    "Fmcnn1": get_fmcnn1,
    "Fmcnn2": get_fmcnn2
}

