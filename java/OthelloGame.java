import java.rmi.Remote;

/**
 * The Othello game controller. The game controller is responsible for
 * the following tasks:
 * <ul>
 *  <li>Telling the players to initialize themselves.</li>
 *  <li>Getting the players' moves.</li>
 *  <li>Ensuring the validity of moves.</li>
 *  <li>(Optional) Ensuring that neither player exceeds the time limit.</li>
 *  <li>(Optional) Communicating the moves to the OthelloObserver.</li>
 * </ul>
 * The game controller can either be started by the user directly, or
 * created by {@link OthelloGameServer} as part of an RMI call from
 * the {@link OthelloTournament} server.
 * <p>
 * $Id: OthelloGame.java,v 1.17 2005/02/17 07:56:11 plattner Exp $
 *
 * @author Aaron Plattner
 **/
public class OthelloGame implements Runnable {
    /** The players. **/
    private OthelloPlayer black, white;

    /** The optional observer. **/
    private OthelloObserver observer;

    /** The game board. **/
    private OthelloBoard board;

    /** True if the game should be timed. **/
    private boolean timed;

    /** The number of milliseconds to give each player.  The time elapsed during
     * a turn is subtracted from the appropriate side's time.
     **/
    private long blackTimeout;
    private long whiteTimeout;

    /**
     * Constructs a new game with the given players.  Players are
     * allowed to think forever in this game.
     *
     * @param black The black player.
     * @param white The white player.
     * @param observer (Optional) The observer.  May optionally be
     * <code>null</code>.
     **/
    public OthelloGame(OthelloPlayer black, OthelloPlayer white, OthelloObserver observer) {
        this.black = black;
        this.white = white;
        this.observer = observer;

        // Players are allowed to think forever.
        timed = false;
        blackTimeout = whiteTimeout = 0;

        board = new OthelloBoard();
    }

    /**
     * Constructs a new game with the given players.  Players are
     * limited to <code>timeout</code> milliseconds per move.
     *
     * @param black The black player.
     * @param white The white player.
     * @param observer (Optional) The observer.  May optionally be
     * <code>null</code>.
     * @param timeout The number of milliseconds each player is allowed
     * to think over the course of the game.
     **/
    public OthelloGame(OthelloPlayer black, OthelloPlayer white, OthelloObserver observer, long timeout) {
        this.black = black;
        this.white = white;
        this.observer = observer;
        // Both players start out with the same amount of time.
        timed = true;
        blackTimeout = whiteTimeout = timeout;

        board = new OthelloBoard();
    }

    /**
     * Runs the game.
     **/
    public void run() {
        OthelloSide turn = OthelloSide.BLACK;
        Move m = null;
        OthelloResult r = new OthelloResult();

        // Initialize the players.
        try {
           doInit(OthelloSide.BLACK);
        } catch(GameException e) {
           // Disqualify them if they fail to init.
           if(observer != null) {
              r.error = e;
              r.conclusion = OthelloResult.BLACK_ERROR_CONCLUSION;
              observer.OnGameOver(r);
           }
           return;
        }

        try {
           doInit(OthelloSide.WHITE);
        } catch(GameException e) {
           // Disqualify them if they fail to init.
           if(observer != null) {
              r.error = e;
              r.conclusion = OthelloResult.BLACK_ERROR_CONCLUSION;
              observer.OnGameOver(r);
           }
           return;
        }

        // Run the game until there are no moves left.
        while(!board.isDone()) {
            // Collect garbage.
            System.gc();

            // Get the player's move.
            long startTime = System.currentTimeMillis();
            try {
               m = getMove(turn, m);
               long endTime = System.currentTimeMillis();

               // Record the player's time and update the time remaining.
               if(turn == OthelloSide.BLACK) {
                  r.blackTime += endTime - startTime;
                  if(timed)
                     blackTimeout -= endTime - startTime;
               } else {
                  r.whiteTime += endTime - startTime;
                  if(timed)
                     whiteTimeout -= endTime - startTime;
               }
            }
            catch(GameException e) {
               // The player had some sort of error, so notify the
               // observer and disqualify them.
               long endTime = System.currentTimeMillis();

               // Record the player's time.
               if(turn == OthelloSide.BLACK)
                  r.blackTime += endTime - startTime;
               else
                  r.whiteTime += endTime - startTime;

               if(observer != null) {
                   // Tell the observer that the game's over.
                   r.error = e;
                   if(turn == OthelloSide.BLACK)
                       r.conclusion = OthelloResult.BLACK_ERROR_CONCLUSION;
                   else
                       r.conclusion = OthelloResult.WHITE_ERROR_CONCLUSION;
                   observer.OnGameOver(r);
               }

               return;
            }

            // Make sure the move is legal.
            // Note that passing is only legal if there are no legal moves.
            if(!board.checkMove(m, turn)) {
                if(observer != null) {
                    // Tell the observer that the game's over.
                    r.error = new BadMoveException(m);
                    if(turn == OthelloSide.BLACK)
                        r.conclusion = OthelloResult.BLACK_ERROR_CONCLUSION;
                    else
                        r.conclusion = OthelloResult.WHITE_ERROR_CONCLUSION;
                    observer.OnGameOver(r);
                }
                return;
            }

            // Notify the observer of the move.
            if(observer != null) {
                observer.OnMove(m, blackTimeout, whiteTimeout);
            }
            // Notify the board of the move.
            board.move(m, turn);

            // It's now the other player's turn.
            if(turn == OthelloSide.BLACK)
                turn = OthelloSide.WHITE;
            else
                turn = OthelloSide.BLACK;
        } /* while */
        // The board is now either full or there are no more legal moves
        // and the game ended normally.

        // Tell the observer that the game's over.
        if(observer != null) {
            r.conclusion = OthelloResult.NORMAL_CONCLUSION;
            r.blackScore = board.countBlack();
            r.whiteScore = board.countWhite();
            // The total running time for these guys is already set.
            observer.OnGameOver(r);
        }
    }

    /**
     * Spawn a thread waiting for a player's move.  If the call doesn't
     * return within the timeout, {@link #getMove} kills it (and
     * anything else in its thread group) and throws TimeoutException.
     *
     * @param turn The side to get the move from.
     * @param lastMove The last player's move.
     * @throws TimeoutException when the call doesn't complete within
     * the specified time.
     * @throws ErrorException when the player has a runtime error while
     * thinking.
     * @return The move the client returned, or <code>null</code> if
     * the client passes on this turn.
     **/
    private Move getMove(final OthelloSide turn, final Move lastMove)
    throws TimeoutException, ErrorException {
        final long timeout = (turn == OthelloSide.BLACK)?blackTimeout:whiteTimeout;
        // If they happen to get exactly 0 milliseconds, prevent them from just
        // taking forever.
        if(timed && timeout <= 0)
           throw new TimeoutException();

        // A wrapper class to spawn a thread to run this function.
        class MoveThread extends Thread implements Runnable {
            public Move m = null;
            // Set if there's a runtime error.
            public Throwable error = null;

            public void run() {
                try {
                    OthelloPlayer p = (turn == OthelloSide.BLACK)?black:white;
                    m = p.doMove(lastMove, timed? timeout : -1);
                }
                catch(Throwable t) {
                    // If there's an error, store it so the player can be
                    // disqualified.
                    error = t;
                }
            }
        }

        // Create a thread group for the thread.
        ThreadGroup g = new ThreadGroup("Player thread group.");
        // Create the thread in the thread group.
        MoveThread t = new MoveThread();
        Thread playerThread = new Thread(g, t);
        playerThread.start();

        // Wait for it to finish
        try {
            playerThread.join(timeout);
        }
        catch(InterruptedException e) {
            // Treat InterruptedExceptions as if they were timeouts.
            throw new TimeoutException();
        }

        // If there's an error, disqualify the player.
        if(t.error != null)
        {
            // Discard the objects to free memory, if necessary.
            black = white = null;
            throw new ErrorException(t.error);
        }

        // If any of the threads are still alive, kill them all and
        // disqualify the player.
        // Get the array of active threads in the thread group.
        // Apparently activeCount() is just an estimate.
        Thread[] threads = new Thread[g.activeCount()];
        g.enumerate(threads);
        for(int i = 0; i < threads.length; i++) {
           if(threads[i] != null && threads[i].isAlive()) {
               // Kill all of the threads in the group (and its
               // subgroups).
               g.stop();
               throw new TimeoutException();
           }
        }

        // Otherwise, return the move.
        return t.m;
    }

    /**
     * Spawn a thread waiting for a player to init.  If the game is timed and
     * the call doesn't return within 30 seconds, {@link #doInit} kills it (and
     * anything else in its thread group) and throws TimeoutException.
     *
     * @param side The side this player is on.
     * @throws TimeoutException when the call doesn't complete within
     * 30 seconds and the game is timed.
     * @throws ErrorException when the player has a runtime error while
     * thinking.
     **/
    private void doInit(final OthelloSide turn)
    throws TimeoutException, ErrorException {
        // A wrapper class to spawn a thread to run this function.
        class InitThread extends Thread implements Runnable {
            // Set if there's a runtime error.
            public Throwable error = null;

            public void run() {
                try {
                    OthelloPlayer p = (turn == OthelloSide.BLACK)?black:white;
                    p.init(turn);
                }
                catch(Throwable t) {
                    // If there's an error, store it so the player can be
                    // disqualified.
                    error = t;
                }
            }
        }

        // Give them a 30 second timeout if timed.
        long timeout = timed?0:30000;

        // Create a thread group for the thread.
        ThreadGroup g = new ThreadGroup("Player thread group.");
        // Create the thread in the thread group.
        InitThread t = new InitThread();
        Thread playerThread = new Thread(g, t);
        playerThread.start();

        // Wait for it to finish
        try {
            playerThread.join(timeout);
        }
        catch(InterruptedException e) {
            // Treat InterruptedExceptions as if they were timeouts.
            throw new TimeoutException();
        }

        // If there's an error, disqualify the player.
        if(t.error != null)
        {
            // Discard the objects to free memory, if necessary.
            black = white = null;
            throw new ErrorException(t.error);
        }

        // If any of the threads are still alive, kill them all and
        // disqualify the player.
        // Get the array of active threads in the thread group.
        // Apparently activeCount() is just an estimate.
        
        if(playerThread.isAlive()) {
            // Kill all of the threads in the group (and its
            // subgroups).
            g.stop();
            throw new TimeoutException();
        }
               
        /*Thread[] threads = new Thread[g.activeCount()];
        g.enumerate(threads);
        for(int i = 0; i < threads.length; i++) {
           if(threads[i].isAlive()) {
               // Kill all of the threads in the group (and its
               // subgroups).
               g.stop();
               throw new TimeoutException();
           }
        }*/
    }
}
