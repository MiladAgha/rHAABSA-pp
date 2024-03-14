from OntologyReasoner import OntReasoner
from config import *

def main(_):

    print('Starting Ontology Reasoner')
    
    Ontology = OntReasoner()
    accuracyOnt, remaining_size = Ontology.run(True, FLAGS.test_path_ont, False)
    print('test acc={:.4f}, remaining size={}'.format(accuracyOnt, remaining_size))

    print('Finished program succesfully')

if __name__ == '__main__':
    tf.app.run()
