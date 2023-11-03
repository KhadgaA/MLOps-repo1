from flask import Flask,request

app = Flask(__name__)

@app.route("/hello/<name>")
def hello_world(name):
    return f"<p>Hello, World! user {name}</p>"

# @app.route("/sum/", method = ["POST"])
# def sum_num():
#     js = request.get_json()
#     x = js["x"]
#     y = js["y"]
#     return f"<p>Hello, user  sum is {x + y}</p>"
