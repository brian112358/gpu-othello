/**
 * Returned in {@link OthelloResult} when a player fails to return
 * from {@link OthelloPlayer#doMove} within the timeout.
 **/
public class TimeoutException extends GameException {
   public String toString() {
      return "Timeout";
   }
}
