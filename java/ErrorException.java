/**
 * Set in {@link OthelloResult} when a player has some sort of runtime
 * exception.
 **/
public class ErrorException extends GameException {
   /** The error thrown by the AI during its turn. **/
   final public Throwable error;

   public ErrorException(final Throwable error) {
     this.error = error;
   }

   public String toString() {
      return "Runtime exception: " + error;
   }
}
