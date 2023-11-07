from flask import Flask,request

app = Flask(__name__)

# @app.route("/hello/<name>")
# def hello_world(name):
#     return f"<p>Hello, World! user {name}</p>"

@app.route("/hello/<a>/<b>")
def hello_world(a,b):
    return f"<p>Hello, World! user {int(a)+int( b)}</p>"

@app.route("/sum/", methods = ["POST"])
def sum_num():
    js = request.get_json()
    x = js["x"]
    y = js["y"]
    return f"<p>Hello, user  sum is {x + y}</p>"
