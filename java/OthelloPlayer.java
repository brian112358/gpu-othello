/**
 * This interface must be implemented by Othello AIs.  Each AI is expected to
 * keep track of the board on its own.  After initialization, the game
 * controller will call {@link #doMove}, passing in the opponent's move, if
 * any along with the time remaining for your AI.
 * <p>
 * The following rules must be followed or your AI will be
 * disqualified:
 * <ul>
 *  <li>{@link #init} must take no longer than 30 seconds.</li>
 *  <li>{@link #doMove} must take no longer than the time remaining for your
 *   side in this game, which will start at 16 minutes of real time for the
 *   tournament and count down as your AI computes.</li>
 *  <li>Your AI must not create threads that continue to run after a call to
 *   either {@link #init} or {@link #doMove}.  Your AI must not try to steal
 *   resources from its opponents in any way.  It is okay to keep multiple
 *   threads around as long as you suspend them before {@link #init} and
 *   {@link #doMove} return.</li>
 *  <li>Finally, your AI must not throw runtime errors such as {@link
 *   java.lang.ArithmeticException}.
 * </ul>
 * <p>
 * $Id: OthelloPlayer.java,v 1.7 2005/02/17 07:56:11 plattner Exp $
 *
 * @author Aaron Plattner
 **/

public interface OthelloPlayer
{
   /**
    * Initialize the AI.
    * @param side Set to either OthelloSide.BLACK or
    * OthelloSide.WHITE.
    **/
   public void init(OthelloSide side);

   /**
    * Compute the next move given the opponent's last move.  Each AI is
    * expected to keep track of the board on its own.  If this is the first
    * move, or the opponent passed on the last move, then
    * <code>opponentsMove</code> will be <code>null</code>.  If there are no
    * valid moves for your side, {@link #doMove} must return <code>null</code>.
    * <p>
    * <strong>Important:</strong> doMove must take no longer than the
    * timeout passed in <tt>millisRemaining</tt>, or your AI will lose!  The
    * move returned must also be legal.
    * <p>
    * You will be disqualified if {@link #doMove} throws any exceptions.
    *
    * @param opponentsMove A {@link Move} object containing the opponent's
    * move, or <code>null</code> if this is the first move or the opponent
    * passed.
    * @param millisLeft The number of milliseconds remaining for your side in
    * the game.
    *
    * @return a {@link Move} object containing your move, or <code>null</code>
    * if you have no valid moves.
    **/
   public Move doMove(Move opponentsMove, long millisLeft);
}
