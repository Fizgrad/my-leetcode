import java.util.HashMap;
public class LFUCache {
    class Item {
        int key;
        int val;
        Item next;
        Item prev;
        int freq = 1;

        Item(int k, int v) {
            key = k;
            val = v;
        }
    }

    class DoublyLinkedList {
        Item head;
        Item tail;

        DoublyLinkedList() {
            head = new Item(-1, -1);
            tail = new Item(-1, -1);
            head.next = tail;
            tail.prev = head;
        }

        void addNode(Item v) {
            Item next = head.next;
            head.next = v;
            v.prev = head;
            v.next = next;
            next.prev = v;
        }

        Item removeNode() {
            Item item = tail.prev;
            item.prev.next = tail;
            tail.prev = item.prev;
            return item;
        }

        Item removeNode(Item v) {
            Item prev = v.prev;
            Item next = v.next;
            prev.next = next;
            next.prev = prev;
            return v;
        }

        boolean isEmpty() {
            return head.next == tail;
        }
    }
    HashMap<Integer, DoublyLinkedList> freqList = new HashMap<>();
    HashMap<Integer, Item> lfuCache = new HashMap<>();
    int capacity;
    int minFreq;

    public LFUCache(int capacity) {
        this.capacity = capacity;
        minFreq = 1;
    }

    public void update(int key) {
        Item v = lfuCache.get(key);
        freqList.get(v.freq).removeNode(v);
        if (freqList.get(v.freq).isEmpty()) {
            if (minFreq == v.freq) {
                minFreq = v.freq + 1;
            }
        }
        v.freq += 1;
        if (!freqList.containsKey(v.freq)) {
            DoublyLinkedList d = new DoublyLinkedList();
            d.addNode(v);
            freqList.put(v.freq, d);
        } else {
            freqList.get(v.freq).addNode(v);
        }
    }

    public int get(int key) {
        if (!lfuCache.containsKey(key))
            return -1;
        update(key);
        return lfuCache.get(key).val;
    }

    public void put(int key, int value) {
        if (capacity == 0)
            return;
        if (lfuCache.containsKey(key)) {
            Item v = lfuCache.get(key);
            update(key);
            v.val = value;
        } else {
            if (lfuCache.size() == capacity) {
                Item v = freqList.get(minFreq).removeNode();
                lfuCache.remove(v.key);
            }
            Item newItem = new Item(key, value);
            lfuCache.put(key, newItem);
            if (freqList.get(1) != null) {
                freqList.get(1).addNode(newItem);
            } else {
                DoublyLinkedList d = new DoublyLinkedList();
                d.addNode(newItem);
                freqList.put(1, d);
            }
            minFreq = 1;
        }
        
    }
}