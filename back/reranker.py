# Compatibility shim: the joblib artifact was serialized with 'reranker' as the module path.
# This re-export keeps deserialization working if the artifact predates models/reranker.py.
from models.reranker import *  # noqa: F401, F403
from models.reranker import TimeDecayReranker  # noqa: F401
