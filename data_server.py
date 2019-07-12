from flask import Flask, request, jsonify
import numpy as np

app = Flask('compare')


DATASET = {}


@app.route('/update', methods=['POST'])
def update():
    identity = request.get_json()
    DATASET.update({identity['id']: identity['feature']})
    return jsonify('done')


def compare(feature1, feature2):
    f1 = np.array(feature1)
    f2 = np.array(feature2)
    return np.dot(f1, f2)/np.linalg.norm(f1)/np.linalg.norm(f2)


MIN_SIMILARITY = .95
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
    if len(id_lst):
        idx = np.argmax(similarity_lst)
        if similarity_lst[idx] > MIN_SIMILARITY:
            ret = id_lst[idx]
    return jsonify(ret)


if __name__ == '__main__':
    app.run('0.0.0.0', 6668, debug=True)
