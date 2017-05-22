import java.io.Serializable;

/**
 * Holds a move, which is simply the (x,y) coordinates of the board position at
 * which a piece is to be placed.  The upper-left corner of the board is (0,0)
 * and the x-axis is to the right.
 * <p>
 * $Id: Move.java,v 1.5 2004/03/13 05:11:29 plattner Exp $
 *
 * @author Aaron Plattner
 **/

public class Move implements Serializable
{
   private int x, y;

   /**
    * Creates a new <code>Move</code> object.
    *
    * @param x The x-coordinate of the move.  Must be non-negative.
    * @param y The y-coordinate of the move.  Must be non-negative.
    * @throws IllegalArgumentException if either of the coordinates is negative.
    **/
   public Move(int x, int y) throws IllegalArgumentException
   {
      if(x<0 || y<0)
         throw new IllegalArgumentException("Positions must be non-negative!");
      this.x = x;
      this.y = y;
   }

   /**
    * Returns the x-coordinate of the move.
    * @return the x-coordinate of the move.
    **/
   public int getX()
   {
      return x;
   }
   
   /**
    * Returns the y-coordinate of the move.
    * @return the y-coordinate of the move.
    **/
   public int getY()
   {
      return y;
   }

   /**
    * Convert a move into a string.
    **/
   public String toString()
   {
      return "(" + x + "," + y + ")";
   }

   /**
    * Compares this move to another object.  Returns false if the
    * object is not a {@link Move}, and true if it is and the
    * coordinates are the same.
    **/
   public boolean equals(Object o)
   {
      if(!(o instanceof Move))
         return false;

      Move m = (Move)o;
      return (x == m.getX() && y == m.getY());
   }

   /**
    * The hashCode of a move is 8*y+x.
    **/
   public int hashCode()
   {
      return 8*y+x;
   }
}
