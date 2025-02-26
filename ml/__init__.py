print("Initializing ml package")
from .model import  MLModelsManagerClassif
from .regres_model import MLModelsManagerRegres
print("MLModelsManager imported successfully")

__all__ = ['MLModelsManagerClassif','MLModelsManagerRegres']
