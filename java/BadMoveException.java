/**
 * Indicates that a player was disqualified due to a bad move.
 **/
public class BadMoveException extends GameException {
   /** The move that disqualified the player. **/
   public final Move move;

   public BadMoveException(final Move m) {
      move = m;
   }

   public String toString() {
      return "Bad move: " + move;
   }
}
