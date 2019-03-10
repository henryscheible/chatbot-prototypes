from trainingsession import TrainingSession
from seq2seqxentropy import seq2seqxentropy
import tools

print("loading data...")
data = tools.load_movie_data()
embedder = Embedder("/data/glove/glove.840B.300d.txt")
