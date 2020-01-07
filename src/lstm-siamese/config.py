EMBEDDING_DIM = 150

MAX_SEQUENCE_LENGTH = 80 # when taking question title + body
VALIDATION_SPLIT = 0.1
RATE_DROP_LSTM = 0.17
RATE_DROP_DENSE = 0.2
NUMBER_LSTM = 50
NUMBER_DENSE_UNITS = 50
ACTIVATION_FUNCTION = 'relu'
MODEL_FILE_NAME = 'SiameseLSTM.h5'

siamese_config = {
	'EMBEDDING_DIM': EMBEDDING_DIM,
	'MAX_SEQUENCE_LENGTH' : MAX_SEQUENCE_LENGTH,
	'VALIDATION_SPLIT': VALIDATION_SPLIT,
	'RATE_DROP_LSTM': RATE_DROP_LSTM,
	'RATE_DROP_DENSE': RATE_DROP_DENSE,
	'NUMBER_LSTM': NUMBER_LSTM,
	'NUMBER_DENSE_UNITS': NUMBER_DENSE_UNITS,
	'ACTIVATION_FUNCTION': ACTIVATION_FUNCTION,
	'MODEL_FILE_NAME': MODEL_FILE_NAME
}