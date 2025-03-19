# Glossary of terms used in this Documentation and in the Geospatial AI area

## Encoder
The neural network used to map between the inputs and the intermdiary stage (usually referred as embedding
or sometimes as latent space) of the forward step. The encoder is also frequently called backbone and, for 
finetuning tasks, it is usually the part of the model which is not updated/trained. 

## Decoder
The neural network employed to map between the intermediary stage (embedding/latent space) and the target
output. For finetuning tasks, the decoder is the most essential part, since it is trained to map the embedding
produced by a previoulsy trained encoder to a new task. 

## Head
A network, usually very small when compared to the encoder and decoder, which is used as final step to adapt
the decoder output to a specific task, for example, by applying a determined activation to it. 

## Neck
Necks are operations placed between the encoder and the decoder stages aimed at adjusting possible
discrepancies, as incompatible shapes, or applying some specific transform, as a normalization required for the task being executed. 

## Factory
A Factory is a class which organizes the instantiation of a complete model, as a backbone-neck-decoder-head
architecture. A class is intended to receive lists and dictionaries containing the required arguments used to
build the model and returns a new instance already ready to be used. 
