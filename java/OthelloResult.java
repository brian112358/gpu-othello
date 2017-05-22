import java.io.Serializable;

/**
 * Stores the result of an Othello game, including the players' scores
 * and what caused the game to end.
 * <p>
 * $Id: OthelloResult.java,v 1.12 2004/03/07 00:20:52 plattner Exp $
 *
 * @author Aaron Plattner
 **/
public class OthelloResult implements Serializable
{
   // Constants for the 'conclusion' field below.
   /** The game ended normally. **/
   public static final int NORMAL_CONCLUSION = 1;
   /** Black was disqualified due to an error of some sort. **/
   public static final int BLACK_ERROR_CONCLUSION = 2;
   /** White was disquialified due to an error of some sort. **/
   public static final int WHITE_ERROR_CONCLUSION = 3;
   /** There was a server error (for example, couldn't load the class file) **/
   public static final int SERVER_ERROR_CONCLUSION = 4;

   /** The player's score. **/
   public int blackScore = 0, whiteScore = 0;
   /** The player's running time, in milliseconds. **/
   public long blackTime = 0, whiteTime = 0;

   /** The reason the game ended. Can take on any of the '_CONCLUSION'
    * values above.
    **/
   public int conclusion = NORMAL_CONCLUSION;

   /** The runtime error that caused the player to be disqualified. **/
   public GameException error;


   /**
    * Determines the winner.
    *
    * @return {@link OthelloSide#BLACK} if black won, {@link
    * OthelloSide#WHITE} if white won, and <code>null</code> if
    * there was a server error.
    **/
   public OthelloSide getWinner()
   {
      // Running out of memory counts as a draw.
      if(error != null && error instanceof ErrorException && ((ErrorException)error).error instanceof OutOfMemoryError)
         return null;

      if( (conclusion == NORMAL_CONCLUSION && blackScore > whiteScore)
        || conclusion == WHITE_ERROR_CONCLUSION)
         return OthelloSide.BLACK;
      else if(
         ( conclusion == NORMAL_CONCLUSION && whiteScore > blackScore)
        || conclusion == BLACK_ERROR_CONCLUSION)
         return OthelloSide.WHITE;
      else
         return null;
   }

   /**
    * Converts an {@link OthelloResult} into a useful string.
    **/
   public String toString()
   {
      if(conclusion == NORMAL_CONCLUSION) {
         //return blackScore + "/" + whiteScore + " (" + blackTime + "ms/" + whiteTime + "ms)";
         String msg = blackScore + "/" + whiteScore;
         if (blackScore > whiteScore) {
            msg += " (Black wins)";
         } else if (blackScore < whiteScore) {
            msg += " (White wins)";
         } else {
            msg += " (Tie)";
         }
         return msg;
      } else if(conclusion == BLACK_ERROR_CONCLUSION) {
         return "Black error: " + error;
      } else if(conclusion == WHITE_ERROR_CONCLUSION) {
         return "White error: " + error;
      } else if(conclusion == SERVER_ERROR_CONCLUSION) {
         // Just print the error because ServerException already
         // prefixes the result with "Server exception:"
         return error.toString();
      } else {
         return "Invalid game conclusion (" + conclusion + ")";
      }
   }
}
