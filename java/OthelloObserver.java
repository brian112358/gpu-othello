/**
 * This interface is implemented by classes that want to watch an
 * Othello game.  The callback methods declared here are called by
 * {@link OthelloGame} when it wants to report moves.  Black always
 * makes the first move.
 * <p>
 * $Id: OthelloObserver.java,v 1.2 2005/02/17 07:56:11 plattner Exp $
 *
 * @author Aaron Plattner
 **/

public interface OthelloObserver
{
   /**
    * Called when a player makes a move.  Note that whoever's playing
    * black always makes the first move.
    *
    * @param m The move the player made.  If the player passed on this
    * turn, <code>m</code> is <code>null</code>.
    * @param blackTimeout Milliseconds remaining for black.
    * @param whiteTimeout Milliseconds remaining for white.
    **/
   public void OnMove(Move m, long blackTimeout, long whiteTimeout);

   /**
    * Called when the game is over.
    *
    * @param result The game result.
    **/
   public void OnGameOver(OthelloResult result);
}
