from flask import Flask, redirect
from flask_wbank import WBank

app = Flask(__name__)
wbank = WBank(app)
app.config["WBANK_API_KEY"] = "xxxxx"
app.config["WBANK_RECIEVER"] = "bangjin"

@app.route("/")
def index():
    return "Hello"
    
@app.route("/wbank/card/payment")
def create_payment():
    if wbank.session:
        return redirect(wbank.session.authPage(url="/process/payment", intent="wbank**/wbank/card/action"))
    return "WBank Session Invaild"
    
@app.route("/process")
@wbank.need_login
def process_pm():
    username, pw, authCode = wbank.session.recvAll()
    pm = wbank.session.create("card-pm", {
      "amount": "9000",
      "username": username,
      "password": pw,
      "authCode": authCode
    })
    req = wbank.request(url="wbank**/wbank/card/action", json=pm, type="wbank**/pay")
    if req.status_code == 200: return "Payment Success"
    return "Payment Failed"