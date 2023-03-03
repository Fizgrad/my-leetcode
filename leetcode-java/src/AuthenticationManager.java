import java.util.HashMap;
import java.util.TreeSet;

public class AuthenticationManager {
    private int timeToLive;
    private TreeSet<Integer> ts;
    private HashMap<String, Integer> hm;
    public AuthenticationManager(int timeToLive) {
        this.timeToLive = timeToLive;
        this.ts = new TreeSet<>();
        this.hm = new HashMap<>();
    }

    public void generate(String tokenId, int currentTime) {
        hm.put(tokenId, currentTime + timeToLive);
        ts.add(currentTime + timeToLive);
    }

    public void renew(String tokenId, int currentTime) {
        Integer time = hm.get(tokenId);
        if (time == null || time <= currentTime) return;
        hm.put(tokenId, currentTime + timeToLive);
        ts.remove(time);
        ts.add(currentTime + timeToLive);
        while (!ts.isEmpty() && ts.lower(currentTime) != null) {
            ts.remove(ts.lower(currentTime));
        }
    }

    public int countUnexpiredTokens(int currentTime) {
        return ts.tailSet(currentTime, false).size();
    }
}
