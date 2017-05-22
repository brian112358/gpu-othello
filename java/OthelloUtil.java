/**
 * Various utility functions.
 * <p>
 * $Id: OthelloUtil.java,v 1.1 2005/02/17 07:56:11 plattner Exp $
 *
 * @author Aaron Plattner
 **/
public class OthelloUtil
{
   /**
    * Converts milliseconds to a minutes/seconds/millis string format.
    *
    * @param ms The number of milliseconds.
    * @return A string representation of <tt>ms</tt>.
    **/
   public static String showTime(long ms)
   {
      long minutes = ms/1000/60;
      long seconds = (ms - 1000*60*minutes)/1000;
      long millis = ms % 1000;
      return minutes + "m " + seconds + "s " + millis +"ms";
   }
}
