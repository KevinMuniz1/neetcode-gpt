import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z -= max(z)
        sumOfNum = round(sum(np.exp(z)),4)
        z = np.exp(z)
        z /= sumOfNum
        return z.round(decimals=4,out=z)       
        
        
    
