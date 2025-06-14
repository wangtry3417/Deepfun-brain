from flask import Flask, redirect
from flask_wbank import WBank
from wbank.Exception import BalanceEnoughException, wbankServerException
from wbank import baseURL

app = Flask(__name__)
wbank = WBank(app)
app.config["WBANK_API_KEY"] = "xxxxx"
app.config["WBANK_RECIEVER"] = "bangjin"

@app.before_request
def test_wbank_server():
    try:
      wbank.request(baseURL)
    except wbankServerException:
     # 請求再次
      wbank.request(baseURL)

@app.route("/")
def index():
    return "Hello"
    
@app.route("/wbank/card/payment")
def create_payment():
    if wbank.session:
        return redirect(wbank.session.authPage(url="/process/payment", intent="wbank**/wbank/card/action"))
    return "WBank Session Invaild"
    
@app.route("/process/payment")
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
    try:
      if req.status_code == 200: return "Payment Success"
      return "Payment Failed"
    except BalanceEnoughException:
      return "Not enough balance"