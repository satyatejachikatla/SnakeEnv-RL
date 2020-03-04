import tensorflow as tf
from SnakeEnv import game_predict_loop
from glob import glob
import sys
if len(sys.argv) < 2:
	model_name = sorted(glob('models/Agent*'))[-1]
else:
	model_name = 'models/'+sys.argv[1]
model = tf.keras.models.load_model(model_name)

print(model_name)
game_predict_loop(model)