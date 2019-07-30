from flask import Flask, request, jsonify
import numpy as np
import pickle


app = Flask('compare')


DATASET = {}
DATASET = pickle.load(open('dataset.pkl', 'rb'))


@app.route('/update', methods=['POST'])
def update():
    identity = request.get_json()
    DATASET.update({identity['id']: identity['feature']})
    return jsonify('done')


def compare(feature1, feature2):
    f1 = np.array(feature1)
    f2 = np.array(feature2)
    cos = np.dot(f1, f2)/np.linalg.norm(f1)/np.linalg.norm(f2)
    return np.exp(cos - 1)


@app.route('/query', methods=['POST'])
def query():
    identity = request.get_json()
    feature = identity['feature']
    similarity_lst = []
    id_lst = []
    for k, v in DATASET.items():
        similarity_lst += [compare(v, feature)]
        id_lst += [k]
    ret = -1
    similarity = 0
    if len(id_lst):
        idx = np.argmax(similarity_lst)
        id = id_lst[idx]
        similarity = similarity_lst[idx]
        ret = id
    return jsonify({'id': ret, 'idx': hash(ret) % 256, 'similarity': similarity})


@app.route('/reset', methods=['POST'])
def reset():
    global DATASET
    # try:
    #     DATASET = pickle.load(open('dataset.pkl', 'rb'))
    # except:
    DATASET = {}
    request.get_json()
    print(len(DATASET), 'saved identities')
    return jsonify(len(DATASET))


@app.route('/save', methods=['POST'])
def save():
    global DATASET
    # pickle.dump(DATASET, open('dataset.pkl', 'wb'))
    DATASET = pickle.load(open('dataset.pkl', 'rb'))
    return jsonify('saved')


if __name__ == '__main__':
    app.run('0.0.0.0', 6669, debug=True)
