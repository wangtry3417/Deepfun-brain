import com.wtech.wbank.Controller;
import com.wtech.wbank.Services;
import java.util.HashMap;
import java.util.Map;

public class wbank {
   public static void main(String args[]) {
      String baseURL = Services.baseURL;
      Controller con = new Controller("api-key");
      Map<String, Object> requestsJson = new HashMap<>();
      con.login();
      Thread.sleep(5000);
      requestsJson.put("cardNumber", "xxxx");
      requestsJson.put("accessKey", con.apiKey);
      requestsJson.put("amount", "1000");
      requestsJson.put("password", "xxxx");
      con.request(baseURL+"/wbank/card/action", requestsJson);
   }
}