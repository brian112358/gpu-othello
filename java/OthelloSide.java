/**
 * Instances of OthelloSide represent the sides in an othello game.
 * There are only two instances, stored in the static fields {@link
 * #BLACK} and {@link #WHITE} of this class.
 * <p>
 * $Id: OthelloSide.java,v 1.8 2005/02/28 10:33:33 plattner Exp $
 **/
public final class OthelloSide {
   /**
    * A private constructor to ensure nobody makes more sides.
    **/
   private OthelloSide() {
   }

   /**
    * Black is a constant used to represent the black side
    **/
   public final static OthelloSide BLACK = new OthelloSide();

   /**
    * White is a constant used to represent the white side
    **/
   public final static OthelloSide WHITE = new OthelloSide();

   /**
   * Returns the opposite of this side.
   **/
   public OthelloSide opposite() {
      if (this == BLACK)
         return WHITE;
      else
         return BLACK;
   }

   /**
   * Returns the opposite of this side.
   **/
   public static OthelloSide opposite(OthelloSide side) {
      if (side == BLACK)
         return WHITE;
      else
         return BLACK;
   }

   /**
    * Returns "Black" if called on {@link OthelloSide#BLACK} and
    * "White" if called on {@link OthelloSide#WHITE}.
    *
    * @throws IllegalArgumentException if called on something other
    * than BLACK or WHITE.
    **/
   public String toString() {
      if(this == BLACK)
         return "Black";
      else if(this == WHITE)
         return "White";
      else
         throw new IllegalArgumentException();
   }
}
