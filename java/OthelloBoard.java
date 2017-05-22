import java.util.BitSet;

/**
 * Implements a standard 8x8 Othello board.
 * <strong>TODO:</strong> Streamline.
 *
 * @author Brandon Moore
 **/
public class OthelloBoard
{
   //locations that have a black piece
   BitSet black = new BitSet(64);
   //locations that have any piece.
   BitSet taken = new BitSet(64);

   public boolean occupied(int x, int y) {
      return taken.get(x + 8*y);
   }

   public boolean get(OthelloSide side, int x, int y) {
      return occupied(x, y) && (black.get(x + 8*y) == (side == OthelloSide.BLACK));
   }

   public void set(OthelloSide side, int x, int y) {
      taken.set(x + 8*y);
      black.set(x + 8*y, side == OthelloSide.BLACK);
   }

   /**
    * Returns a copy of this board.
    **/
   public OthelloBoard copy() {
      OthelloBoard newBoard = new OthelloBoard();
      newBoard.black = (BitSet)this.black.clone();
      newBoard.taken = (BitSet)this.taken.clone();

      return newBoard;
   }

   /**
    * Make a standard 8x8 othello board.
    * Initialize to the standard setup.
    **/
   public OthelloBoard() {
      //Standard setup with 4 pieces in the center.
      taken.set(3 + 8 * 3);
      taken.set(3 + 8 * 4);
      taken.set(4 + 8 * 3);
      taken.set(4 + 8 * 4);
      black.set(4 + 8 * 3);
      black.set(3 + 8 * 4);
   }

   /**
    * Tests if the game is finished. The game is finished if neither
    * side has a legal move.
    * @return true if there are no legal moves.
    **/
   public boolean isDone() {
      return !(hasMoves(OthelloSide.BLACK) || hasMoves(OthelloSide.WHITE));
   }

   /**
    * Tests for legal moves.
    * @param side Othelloside to check for valid moves.
    * @return true if there are legal moves.
    **/
   public boolean hasMoves(OthelloSide side) {
      for (int i = 0; i < 8; i++) {
         for (int j = 0; j < 8; j++) {
            if (checkMove(new Move(i, j), side)) {
               return true;
            }
         }
      }
      return false;
   }

   boolean onBoard(int x, int y) {
      return(0 <= x && x < 8 && 0 <= y && y < 8);
   }

   /**
    * Tests if a move is legal.
    * @param m The move being made
    * @param turn The player making the move.
    **/
   //might be able to do clever stuff with masks and next clear bit
   //and the like.
   public boolean checkMove(Move m, OthelloSide turn) {
      if(m == null)
         //passing is only legal if you have no moves
         return !hasMoves(turn);

      // Make sure the square hasn't already been taken.
      if(occupied(m.getX(), m.getY()))
         return false;

      OthelloSide other = turn.opposite();
      int X = m.getX();
      int Y = m.getY();
      for (int dx = -1; dx <= 1; dx++) {
         for (int dy = -1; dy <= 1; dy++) {
            //for each direction
            if (dy == 0 && dx == 0)
               continue;

            //is there a capture in that direction?
            int x = X + dx;
            int y = Y + dy;
            if (onBoard(x,y) && get(other,x,y)) {
               do {
                  x += dx;
                  y += dy;
               } while (onBoard(x,y) && get(other,x,y));
               if (onBoard(x,y) && get(turn,x,y)) {
                  return true;
               }
            }
         }
      }
      return false;
   }

   /**
    * Modifies the board to reflect the specified move.
    * @param m The move being made
    * @param turn The player making the move
    **/
   public void move(Move m, OthelloSide turn) {
      // null means pass.
      if (m == null)
        return;

      if (!checkMove(m, turn)) {
            throw new InternalError("Invalid Move " + m);
      }

      OthelloSide other = turn.opposite();
      for (int dx = -1; dx <= 1; dx++) {
         for (int dy = -1; dy <= 1; dy++) {
            if (dy == 0 && dx == 0) {
               continue;
            }
            int x = m.getX();
            int y = m.getY();
            do {
               x += dx;
               y += dy;
            } while (onBoard(x,y) && get(other,x,y));
            if (onBoard(x,y) && get(turn,x,y)) {
               x = m.getX();
               y = m.getY();
               x += dx;
               y += dy;
               while (onBoard(x,y) && get(other,x,y)) {
                  set(turn,x,y);
                  x += dx;
                  y += dy;
               }
            }
         }
      }
      set(turn,m.getX(),m.getY());
   }

   /**
    * Current count of black stones.
    * @return The number of black stones on the board.
    **/
   public int countBlack() {
      return black.cardinality();
   }

   /**
    * Current count of white stones.
    * @return The number of white stones on the board.
    **/
   public int countWhite() {
      BitSet result = (BitSet)taken.clone();
      result.andNot(black);
      return result.cardinality();
   }
}
