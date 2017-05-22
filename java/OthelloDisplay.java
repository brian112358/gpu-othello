import java.awt.*;
import java.awt.event.*;
import javax.swing.*;

/**
 * Displays an Othello game in a window.  This class can either be
 * used as an {@link OthelloPlayer} or as an {@link OthelloObserver}.
 * Trying to use the same object as both a player and an observer
 * results in a {@link IllegalStateException}.
 * <p>
 * $Id: OthelloDisplay.java,v 1.18 2005/02/17 07:56:11 plattner Exp $
 *
 * @author Aaron Plattner
 * Some code stolen from MissileCommandApplication.java by Joseph
 * Gonzalez.
 **/
public class OthelloDisplay
   extends JFrame
   implements OthelloObserver, OthelloPlayer, ActionListener
{
   // The roles an OthelloDisplay can perform.
   private final static int PLAYER_ROLE = 1;
   private final static int OBSERVER_ROLE = 2;
   /** The role of this OthelloObserver. */
   private int role = 0;
   /** A label that describes the current state of the game. */
   private JLabel stateLabel;
   /** Determines who's turn it is. **/
   private OthelloSide turn = OthelloSide.BLACK;
   /** Our copy of the board. **/
   private OthelloBoard board;
   /** The component that displays the board. **/
   private BoardComponent boardComponent;
   /** The names of the other players, if given. **/
   private String blackName = null, whiteName = null;

   //// OthelloPlayer data.
   /**
    * Our move to be returned from doMove.
    * The click handler sets this and then calls moveLock.notify().
    * doMove blocks on this.  That way, the click-handling code
    * doesn't really need to know whether doMove is pending or not,
    * and doMove just needs to wait.
    **/
   private Move ourMove;
   /**
    * The object that threads wishing to access ourMove should
    * synchronize on.
    **/
   private final Object moveLock;
   /** Our side of the battle. **/
   private OthelloSide ourSide;

   /**
    * Creates and shows a board window.  When the window is closed,
    * the application exits.
    **/
   public OthelloDisplay(int which)
   {
      super("Othello");
      JPanel screen = new JPanel(new BorderLayout());
      board = new OthelloBoard();
      boardComponent = new BoardComponent(board);

      // Throw the board into the screen.
      screen.add(boardComponent, BorderLayout.NORTH);
      stateLabel = new JLabel("Waiting");
      screen.add(stateLabel, BorderLayout.SOUTH);
      getContentPane().add(screen);

      // Register ourselves as the board's ActionListener
      boardComponent.addActionListener(this);

      if(which != 0)
      {
         int xloc = (BoardComponent.canvasSize + 16) * which;
         int yloc = 0;
         this.setLocation(xloc,yloc);
      }

      setResizable(true);
      setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      pack();
      setVisible(true);

      moveLock = new Object();

      blackName = "Black";
      whiteName = "White";
   }

   public OthelloDisplay()
   {
      this(0);
   }

   /**
    * Prevents the frame from killing the VM when it's closed.
    **/
   public void disableClose()
   {
      setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
   }

   /**
    * A constructor that takes the player names and displays them in
    * the title.
    **/
   public OthelloDisplay(String blackName, String whiteName)
   {
      this(blackName,whiteName,0);
   }
   
   public OthelloDisplay(String blackName, String whiteName, int which)
   {
      this(which);

      this.blackName = blackName + " (black)";
      this.whiteName = whiteName + " (white)";
      setTitle(blackName + " vs. " + whiteName);
   }

   protected void handleMove(Move m)
   {
      // Update the board and repaint the board component.
      board.move(m, turn);
      // 10-ms update?
      boardComponent.repaint(10, 0,0,boardComponent.canvasSize, boardComponent.canvasSize);

      turn = turn.opposite();
   }

   public void OnMove(Move m, long blackTimeout, long whiteTimeout)
   {
      // Check our role
      if(role == 0)
      {
         role = OBSERVER_ROLE;
      } else if(role != OBSERVER_ROLE)
         throw new IllegalStateException("Can't use the same OthelloDisplay as both an observer and a player!");

      handleMove(m);

      setTitle("Observing (" + blackName + ": " + board.countBlack() + " " + whiteName + ": " + board.countWhite() + ")" );

      // Update the label.
      if(turn == OthelloSide.BLACK)
         stateLabel.setText(blackName + "'s turn.  Time left: " + OthelloUtil.showTime(blackTimeout) + "  Other move: " + m);
      else
         stateLabel.setText(whiteName + "'s turn.  Time left: " + OthelloUtil.showTime(whiteTimeout) + "  Other move: " + m);
   }

   public void OnGameOver(OthelloResult r)
   {
      setTitle("Observing " + blackName + " vs. " + whiteName);
      stateLabel.setText("Game over!  Result: " + r);
   }

   public void init(OthelloSide side)
   {
      // Check our role
      if(role == 0)
      {
         role = PLAYER_ROLE;
      } else
         throw new IllegalStateException("init called with role != 0 (role = " + role + ")");

      ourSide = side;
      // The other player goes "first", although their move may be null.
      turn = ourSide.opposite();
      setTitle(side.toString());
   }

   public Move doMove(Move otherMove, long millisLeft)
   {
      handleMove(otherMove);
      stateLabel.setText("Our turn, other move was " + otherMove + " Time left: " + OthelloUtil.showTime(millisLeft));

      // Pass if no valid moves.
      if(!board.hasMoves(ourSide))
      {
         stateLabel.setText("No moves.  Passing.");
         handleMove(null);
         return null;
      }

      // Wait for the frontend to give us a move.
      synchronized(moveLock)
      {
         ourMove = null;

         // Wait for the move to be valid.
         while(ourMove == null || !board.checkMove(ourMove, ourSide))
         {
            if(ourMove != null)
            {
               stateLabel.setText("Invalid move: " + ourMove + " ... try again!");
            }

            try {
               moveLock.wait();
            } catch(InterruptedException e) {
               System.out.println("OthelloDisplay error: Interrupted during doMove!");
               return null;
            }
         }

         // Handle our own move.
         handleMove(ourMove);
         stateLabel.setText("Opponent's turn.");
         return ourMove;
      }

   }

   public void actionPerformed(ActionEvent e)
   {
      // The user clicked somewhere.
      if(e.getSource() == boardComponent && e instanceof MoveEvent)
      {
         synchronized(moveLock)
         {
            // Set our move and notify the doMove thread, if it's
            // listening.
            ourMove = ((MoveEvent)e).getMove();
            moveLock.notifyAll();
         }
      }
   }
}

class MoveEvent extends ActionEvent
{
   private final Move m;

   public MoveEvent(Object source, Move m)
   {
      super(source, 0, "Move");
      this.m = m;
   }

   public Move getMove() { return m; }
}

/**
 * Displays an OthelloBoard.
 **/
class BoardComponent extends JPanel
{
   public static final int squareSize = 48;
   public static final int canvasSize = squareSize*8;

   // TODO: Come up with a better way of doing this.
   protected ActionListener moveListener = null;

   /** Our reference to the board. */
   private OthelloBoard board;

   public BoardComponent(OthelloBoard b)
   {
      board = b;

      setPreferredSize(new Dimension(canvasSize, canvasSize));
      enableEvents(MouseEvent.MOUSE_CLICKED);
   }

   public void addActionListener(ActionListener l)
   {
      moveListener = l;
   }

   /** Draw the contents of the board. */
   public void paintComponent(Graphics graphics)
   {
      Graphics2D g = (Graphics2D)graphics.create();
      g.setRenderingHint
        (RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

      // Color the background.
      g.setColor(new Color(0, 153, 0));
      g.fillRect(0, 0, getWidth(), getHeight());

      // Draw the grid.
      g.setColor(new Color(0, 0, 0));
      for(int i=1; i<=8; i++)
      {
         int x = i*squareSize;
         g.drawLine(x,0,x,8*squareSize);
         g.drawLine(0,x,8*squareSize,x);
      }

      // Draw the pieces.
      for(int i=0; i<8; i++)
      {
         for(int j=0; j<8; j++)
         {
            final int x = i*squareSize;
            final int y = j*squareSize;

            if(board.get(OthelloSide.BLACK, i,j))
            {
               g.setColor(Color.black);
               g.fillOval(x+2,y+2, squareSize-4, squareSize-4);
            } else if(board.get(OthelloSide.WHITE, i,j))
            {
               g.setColor(Color.white);
               g.fillOval(x+2,y+2, squareSize-4, squareSize-4);
            }

         }
      }
   }

   protected void processMouseEvent(MouseEvent e)
   {
      if(e.getID() == MouseEvent.MOUSE_CLICKED)
      {
         final int x = e.getX();
         final int y = e.getY();
         final int i = x/squareSize;
         final int j = y/squareSize;

         if(moveListener != null)
            moveListener.actionPerformed(new MoveEvent(this, new Move(i,j)));
      }
   }
}
