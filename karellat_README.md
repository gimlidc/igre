 # Improvement of visibility of concealed features in artwork NIR reflectograms by information separation
 ## IG
Transfer function f: I(VIS) → I(NIR)
k classes - set of pixels with similar optical properties
Cᵢ - set of pixels of material i (== same color, same material)
pᵢₒ - pixel of class Cᵢ 
cᵢ - mean value of Cᵢ; n(cᵢ, NIR, o)
n(cᵢ, NIR, o) - distance of pᵢₒ and cᵢ (deviation)
I(pᵢₒ, NIR) - reflectance intesity measured in NIR


T = {C₁, C₂ ... Cₖ} 
j = |Cᵢ|; Cᵢ = {pᵢ₁, pᵢ₂ ... pᵢⱼ }
I(pᵢₒ, NIR) = I(cᵢ, NIR) + n(cᵢ, NIR, d)
f:I(pᵢₒ, VIS) + n(cᵢ, VIS, o) → I(pᵢₖ, NIR) + n(cᵢ, NIR, o) 

Estimation of f by fₜ
fₜ (Iᵥᵢₛ) = Îₙᵢᵣ. Δ = |Iₙᵢᵣ - Îₙᵢᵣ|.

fₜ should predict as much as possible the Iₙᵢᵣ, but based only on information from Iᵥᵢₛ. The resulting Îₙᵢᵣ is the closest estimation of Iₙᵢᵣ based on visible spectrum, therefore the information hidden in NIR spectrum could be revealed simply by substracting the Îₙᵢᵣ from the Iₙᵢᵣ.

Function the fₜ: 
INPUTS: 
    pixel of the Iᵥᵢₛ (several channels)
OUTPUTS:
    single channel NIR pixel
LOSS: 
    MSE(fₜ(Iᵥᵢₛ),Iₙᵢᵣ)  
