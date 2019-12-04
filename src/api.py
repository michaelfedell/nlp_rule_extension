from flask import Flask
from flask_restful import Resource, Api, reqparse
import argparse

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('text', type=str, help='Raw text of review for prediction')


class Check(Resource):
    @staticmethod
    def get():
        return {'version': '0.0', 'metrics': 'some performance metrics'}


class Predict(Resource):
    @staticmethod
    def post():
        args = parser.parse_args()
        return args['text'], 200


api.add_resource(Check, '/')
api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True)
