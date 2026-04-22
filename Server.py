import requests
from datetime import datetime
import time
import threading
from collections import deque
import os
import copy

MONGO_OK = False
try:
    from pymongo import MongoClient
    MONGO_OK = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════════
# GAME RULES
# ══════════════════════════════════════════════════════════════════════
class Rules:
    @staticmethod
    def size(n):   return 'B' if n >= 5 else 'S'
    @staticmethod
    def color(n):
        if n in [1,3,5,7,9]: return 'G'
        if n in [2,4,6,8]:   return 'R'
        return 'V'  # 0 and 5 are violet (5 counts green too)
    @staticmethod
    def color_wins(pred, n):
        # violet counts as both R and G
        if n == 0: return pred in ['R','V']
        if n == 5: return pred in ['G','V']
        return pred == Rules.color(n)
    @staticmethod
    def size_full(n): return 'big' if n >= 5 else 'small'
    @staticmethod
    def color_full(c):
        return {'G':'green','R':'red','V':'violet'}.get(c,'red')

# ══════════════════════════════════════════════════════════════════════
# MONGODB
# ══════════════════════════════════════════════════════════════════════
class DB:
    def __init__(self):
        self.ok = False
        self.writes = 0
        if not MONGO_OK: return
        try:
            import dns.resolver
            r = dns.resolver.Resolver()
            r.nameservers = ['8.8.8.8']
            dns.resolver.default_resolver = r
        except: pass
        try:
            uri = ("mongodb+srv://krishnavishwas011_db_user:OgktWrNR3KGzo2rj@"
                   "datacenter.xuicoag.mongodb.net/ai_predictions"
                   "?retryWrites=true&w=majority&appName=Datacenter")
            self.client = MongoClient(uri, serverSelectionTimeoutMS=6000)
            self.db = self.client['ai_predictions']
            self.ok = True
        except: pass

    def save(self, col, doc):
        if not self.ok: return
        try:
            self.db[col].insert_one(doc)
            self.writes += 1
        except: pass

    def upsert(self, col, filt, doc):
        if not self.ok: return
        try:
            self.db[col].replace_one(filt, doc, upsert=True)
            self.writes += 1
        except: pass

    def status(self): return f"DB:OK({self.writes})" if self.ok else "DB:OFF"

# ══════════════════════════════════════════════════════════════════════
# CYCLE DETECTOR  — learns patterns like BBSBBSSS or RRGGRRRG live
# ══════════════════════════════════════════════════════════════════════
class CycleDetector:
    """
    Scans history for repeating sub-sequences.
    Tries cycle lengths 2..8.
    Picks the cycle with highest recent match rate.
    After every result it re-evaluates — fully live.
    """
    MIN_LEN = 2
    MAX_LEN = 8
    LOOK_BACK = 40   # how many recent values to score cycles on

    def __init__(self):
        self.hist = deque(maxlen=500)   # e.g. 'B','S','B',...

    def push(self, val):
        self.hist.appendleft(val)   # newest at index 0

    def _score_cycle(self, seq, h):
        """How well does repeating `seq` predict values in h (newest first)?"""
        n = len(seq)
        hits = 0
        total = 0
        for i in range(len(h) - 1):
            pos_in_cycle = i % n
            pred = seq[pos_in_cycle]
            actual = h[i]
            if pred == actual:
                hits += 1
            total += 1
        return hits / total if total else 0

    def best_cycle(self):
        h = list(self.hist)
        if len(h) < 10:
            return None, 0, 0

        lb = h[:self.LOOK_BACK]
        best_seq   = None
        best_score = 0
        best_len   = 0

        for length in range(self.MIN_LEN, self.MAX_LEN + 1):
            if len(h) < length * 3:
                continue
            # Candidate: the most recent `length` values reversed (oldest first = cycle template)
            # We try the cycle as it appears oldest-first
            candidate = list(reversed(h[:length]))
            score = self._score_cycle(candidate, lb)
            if score > best_score:
                best_score = score
                best_seq   = candidate
                best_len   = length

        return best_seq, best_score, best_len

    def predict(self):
        """
        Returns (next_val, confidence, cycle_str, score)
        next_val: 'B'/'S' or 'G'/'R'/'V'
        """
        h = list(self.hist)
        if len(h) < 8:
            return None, 0, '', 0

        seq, score, length = self.best_cycle()
        if seq is None or score < 0.55:
            return None, 0, '', score

        # Current position in cycle = index 0 in hist corresponds to position 0
        # hist[0] = just played, hist[1] = before that, etc.
        # Position in cycle for the NEXT value:
        # hist[0] was at position (length-1) if cycle aligns perfectly,
        # but we detect actual position by finding best alignment offset
        best_offset = 0
        best_align  = -1
        for offset in range(length):
            hits = 0
            for i in range(min(len(h), self.LOOK_BACK)):
                expected = seq[(i + offset) % length]
                if expected == h[i]:
                    hits += 1
            if hits > best_align:
                best_align  = hits
                best_offset = offset

        next_pos  = (best_offset - 1) % length   # one step ahead of current
        next_val  = seq[next_pos]
        conf      = int(score * 100)
        cycle_str = ''.join(seq)

        return next_val, conf, cycle_str, score

# ══════════════════════════════════════════════════════════════════════
# SIMPLE STREAK / MAJORITY  (fallback when no cycle)
# ══════════════════════════════════════════════════════════════════════
class SimpleAnalyzer:
    def __init__(self):
        self.hist = deque(maxlen=500)

    def push(self, val):
        self.hist.appendleft(val)

    def predict(self):
        h = list(self.hist)
        if len(h) < 4:
            return None, 0, ''

        # Streak length from front
        sl = 1
        for i in range(1, len(h)):
            if h[i] == h[0]: sl += 1
            else: break

        # After long streak (>=5), expect flip
        if sl >= 5:
            opp = self._opp(h[0])
            return opp, min(82, 65 + sl * 2), f"FLIP after {sl}x{h[0]}"

        # AABB: h[0]==h[1] != h[2]==h[3] → next = h[2]
        if len(h) >= 4 and h[0]==h[1] and h[2]==h[3] and h[0]!=h[2]:
            return h[2], 78, f"AABB→{h[2]}"

        # AAABBB
        if len(h) >= 6 and h[0]==h[1]==h[2] and h[3]==h[4]==h[5] and h[0]!=h[3]:
            return h[0], 80, f"AAABBB cont {h[0]}"

        # ABAB alternating
        if len(h) >= 4 and h[0]!=h[1] and h[0]==h[2] and h[1]==h[3]:
            return self._opp(h[0]), 76, f"ABAB→{self._opp(h[0])}"

        # Majority last 6
        last6 = h[:6]
        cnt = {}
        for x in last6: cnt[x] = cnt.get(x,0)+1
        dom = max(cnt, key=cnt.get)
        if cnt[dom] >= 4:
            return dom, 70, f"MAJ {cnt[dom]}/6 {dom}"

        return None, 0, ''

    @staticmethod
    def _opp(v):
        return {'B':'S','S':'B','G':'R','R':'G','V':'G'}.get(v, v)

# ══════════════════════════════════════════════════════════════════════
# ADAPTIVE SCORER  — tracks per-method accuracy live
# ══════════════════════════════════════════════════════════════════════
class AdaptiveScorer:
    def __init__(self):
        # method → [wins, total]
        self.rec = {'cycle':[0,0], 'simple':[0,0], 'flip':[0,0]}

    def hit(self, method):
        m = self._key(method)
        self.rec[m][0] += 1
        self.rec[m][1] += 1

    def miss(self, method):
        m = self._key(method)
        self.rec[m][1] += 1

    def wr(self, method):
        m = self._key(method)
        t = self.rec[m][1]
        return round(self.rec[m][0]/t*100, 0) if t else 50.0

    def _key(self, method):
        for k in self.rec:
            if k in method: return k
        return 'simple'

    def display(self):
        return "  ".join([f"{m}:{self.rec[m][0]}/{self.rec[m][1]}({self.wr(m):.0f}%)"
                          for m in self.rec])

# ══════════════════════════════════════════════════════════════════════
# MASTER PREDICTOR  — ties everything together, self-updates live
# ══════════════════════════════════════════════════════════════════════
class Predictor:
    def __init__(self, ptype):
        self.ptype   = ptype           # 'size' or 'color'
        self.cycle   = CycleDetector()
        self.simple  = SimpleAnalyzer()
        self.scorer  = AdaptiveScorer()
        self.seen    = set()
        self.total   = 0
        self.consec_loss = 0
        self.flip_active = False
        self.recent  = deque(maxlen=30)

    # ── ingest ───────────────────────────────────────────────────────
    def ingest(self, number, period=None):
        if period and period in self.seen: return False
        if period: self.seen.add(period)
        val = Rules.size(number) if self.ptype=='size' else Rules.color(number)
        self.cycle.push(val)
        self.simple.push(val)
        self.total += 1
        return True

    def bulk(self, history):
        for item in reversed(history):
            self.ingest(item.get('number'), item.get('period'))

    # ── feedback ─────────────────────────────────────────────────────
    def feedback(self, is_win, method):
        self.recent.append(1 if is_win else 0)
        if is_win:
            self.scorer.hit(method)
            self.consec_loss = 0
            self.flip_active = False
        else:
            self.scorer.miss(method)
            self.consec_loss += 1
            if self.consec_loss >= 3:
                self.flip_active = True

    def winrate(self):
        r = list(self.recent)
        return round(sum(r)/len(r)*100,1) if r else 0.0

    # ── predict ──────────────────────────────────────────────────────
    def predict(self):
        if self.total < 8:
            val = 'B' if self.ptype=='size' else 'G'
            return self._pack(val, 50, 'loading', f'Loading {self.total}/8')

        # Try cycle first
        cv, cc, cstr, cscore = self.cycle.predict()
        if cv and cc >= 58:
            method = 'cycle'
            val, conf = cv, cc
            reason = f"Cycle [{cstr}] score={cscore:.0f}%"
        else:
            # Fallback to simple
            sv, sc, sreason = self.simple.predict()
            if sv and sc >= 60:
                method = 'simple'
                val, conf = sv, sc
                reason = sreason
            else:
                # Last resort: follow current cycle value or majority
                h = list(self.cycle.hist)
                val = h[0] if h else ('B' if self.ptype=='size' else 'G')
                conf = 55
                method = 'follow'
                reason = f"Following {val}"

        # FLIP: after 3 straight losses, invert
        if self.flip_active:
            val    = self._opp(val)
            conf   = min(conf + 8, 92)
            method = 'flip_' + method
            reason = f"[FLIP/{self.consec_loss}L] {reason}"

        return self._pack(val, conf, method, reason)

    def _pack(self, raw, conf, method, reason):
        if self.ptype == 'size':
            value = 'big' if raw == 'B' else 'small'
        else:
            value = Rules.color_full(raw)
        return {'value': value, 'raw': raw, 'conf': conf,
                'method': method, 'reason': reason}

    @staticmethod
    def _opp(v):
        return {'B':'S','S':'B','G':'R','R':'G','V':'G'}.get(v, v)

# ══════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self, pred, vtype):
        self.pred    = pred
        self.vtype   = vtype
        self.pending = None
        self.pper    = None
        self.wins = self.losses = 0
        self.wstrk = self.lstrk = 0
        self.last = None

    def set(self, p, period):
        self.pending = copy.deepcopy(p)
        self.pper    = period

    def verify(self, item):
        if self.pending is None: return None
        p, per = self.pending, self.pper
        self.pending = None; self.pper = None
        n = item.get('number', -1)

        if self.vtype == 'size':
            is_win = p['value'] == Rules.size_full(n)
        else:
            is_win = Rules.color_wins(p['raw'], n)

        if is_win:
            self.wins += 1; self.wstrk += 1; self.lstrk = 0
        else:
            self.losses += 1; self.lstrk += 1; self.wstrk = 0

        self.pred.feedback(is_win, p['method'])

        self.last = {
            'period': per, 'pred': p['value'], 'raw': p['raw'],
            'method': p['method'], 'is_win': is_win,
            'actual_n': n,
            'actual_s': Rules.size_full(n),
            'actual_c': item.get('color', Rules.color_full(Rules.color(n)))
        }
        return self.last

    def wr(self):
        t = self.wins + self.losses
        return round(self.wins/t*100,1) if t else 0.0

# ══════════════════════════════════════════════════════════════════════
# DUAL SERVER
# ══════════════════════════════════════════════════════════════════════
class Server:
    URL30 = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    URL1M = "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json"

    def __init__(self):
        self.db   = DB()
        self.lock = threading.Lock()

        self.dark = Predictor('color')
        self.fox  = Predictor('size')

        self.dk30 = Tracker(self.dark, 'color')
        self.dk1m = Tracker(self.dark, 'color')
        self.fx30 = Tracker(self.fox,  'size')
        self.fx1m = Tracker(self.fox,  'size')

        self.g = {
            '30s': {'hist':[], 'last_per':None,
                    'dkp':None,'dkt':None, 'fxp':None,'fxt':None},
            '1m':  {'hist':[], 'last_per':None,
                    'dkp':None,'dkt':None, 'fxp':None,'fxt':None},
        }
        self.inited = {'30s':False,'1m':False}

    # ── extract ──────────────────────────────────────────────────────
    def extract(self, raw):
        recs = []
        if isinstance(raw, dict):
            d = raw.get('data', {})
            if isinstance(d, dict):
                for k in ['results','list','rows','items']:
                    if k in d and isinstance(d[k], list):
                        recs = d[k]; break
            if not recs:
                for k in ['data','result','list','rows']:
                    if k in raw and isinstance(raw[k], list):
                        recs = raw[k]; break
        elif isinstance(raw, list):
            recs = raw

        out = []
        for rec in recs:
            if not isinstance(rec, dict): continue
            item = {}
            for k in ['issue_number','issueNumber','issue','period']:
                if k in rec and rec[k]:
                    item['period'] = str(rec[k]); break
            for k in ['result_number','number','num','openNum']:
                if k in rec and rec[k] is not None:
                    try:
                        item['number'] = int(str(rec[k]).split('=')[-1]) % 10
                        break
                    except: continue
            if 'number' not in item: continue
            n = item['number']
            item['size']  = Rules.size_full(n)
            item['color'] = Rules.color_full(Rules.color(n))
            if 'big_small' in rec:
                item['size'] = 'big' if str(rec['big_small']).upper()=='BIG' else 'small'
            out.append(item)
        return out

    def fetch(self, url):
        try:
            r = requests.get(url, timeout=8, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code == 200: return r.json()
        except: pass
        return None

    # ── process ──────────────────────────────────────────────────────
    def process(self, raw, game):
        data = self.extract(raw)
        if not data: return

        g = self.g[game]
        cur = data[0].get('period')
        if cur == g['last_per']: return
        g['last_per'] = cur

        # Merge
        existing = {h.get('period') for h in g['hist']}
        new = [x for x in data if x.get('period') not in existing]
        for x in reversed(new):
            g['hist'].insert(0, x)
        g['hist'] = g['hist'][:1000]

        dkt = self.dk30 if game=='30s' else self.dk1m
        fxt = self.fx30 if game=='30s' else self.fx1m
        latest = data[0]

        # Bulk init
        if not self.inited[game]:
            self.dark.bulk(g['hist'])
            self.fox.bulk(g['hist'])
            self.inited[game] = True
        else:
            for x in reversed(new):
                self.dark.ingest(x['number'], x.get('period'))
                self.fox.ingest(x['number'], x.get('period'))

        # Verify old predictions
        dk_r = dkt.verify(latest)
        fx_r = fxt.verify(latest)

        # Save results to DB
        now_ts  = datetime.now()
        now_iso = now_ts.isoformat()
        now_str = now_ts.strftime('%Y-%m-%d %H:%M:%S')

        if dk_r:
            threading.Thread(target=self.db.save, args=(
                f"dark_server_history_{game}", {
                    'period':      dk_r['period'],
                    'pred':        dk_r['pred'],
                    'prediction':  dk_r['pred'].upper(),
                    'actual':      str(dk_r['actual_n']),
                    'actual_color':dk_r['actual_c'],
                    'actual_size': dk_r['actual_s'],
                    'win':         dk_r['is_win'],
                    'result':      '✅' if dk_r['is_win'] else '❌',
                    'method':      dk_r['method'],
                    'server':      'DARK SERVER',
                    'game':        game,
                    'ts':          now_iso,
                    'created_at':  now_str,
                }), daemon=True).start()

        if fx_r:
            threading.Thread(target=self.db.save, args=(
                f"pro_fox_history_{game}", {
                    'period':      fx_r['period'],
                    'pred':        fx_r['pred'],
                    'prediction':  fx_r['pred'].upper(),
                    'actual':      str(fx_r['actual_n']),
                    'actual_color':fx_r['actual_c'],
                    'actual_size': fx_r['actual_s'],
                    'win':         fx_r['is_win'],
                    'result':      '✅' if fx_r['is_win'] else '❌',
                    'method':      fx_r['method'],
                    'server':      'PRO FOX',
                    'game':        game,
                    'ts':          now_iso,
                    'created_at':  now_str,
                }), daemon=True).start()

        # New predictions
        target = str(int(cur) + 1) if cur else '?'

        dkp = self.dark.predict()
        fxp = self.fox.predict()
        g['dkp'] = dkp; g['dkt'] = target
        g['fxp'] = fxp; g['fxt'] = target
        dkt.set(dkp, target)
        fxt.set(fxp, target)

        # Save latest — exact format API expects
        for p, server_code, server_name in [
            (dkp, 'DARK_SERVER', 'DARK v11'),
            (fxp, 'PRO_FOX',     'FOX v11'),
        ]:
            ts_now = datetime.now()
            doc = {
                # top-level fields API reads
                'server':    server_code,
                'game':      game,
                'type':      'latest',
                'success':   True,
                'timestamp': ts_now.isoformat(),
                # nested data block — exactly what API returns
                'data': {
                    'confidence':    f"{p['conf']}%",
                    'game':          game,
                    'prediction':    p['value'].upper(),
                    'server':        server_name,
                    'target_period': target,
                    'updated_at':    ts_now.strftime('%Y-%m-%d %H:%M:%S'),
                },
                # also flat for easy querying
                'pred':   p['value'],
                'conf':   p['conf'],
                'method': p['method'],
                'target': target,
                'ts':     ts_now.isoformat(),
            }
            col_name = f"{server_code.lower()}_latest_{game}"
            filt     = {'server': server_code, 'game': game}
            threading.Thread(target=self.db.upsert,
                             args=(col_name, filt, doc),
                             daemon=True).start()

        self.draw()

    # ── draw ─────────────────────────────────────────────────────────
    def draw(self):
        with self.lock: self._draw()

    def _draw(self):
        os.system('cls' if os.name=='nt' else 'clear')
        now = datetime.now()
        sec = now.second
        n30 = 30-sec if sec<30 else 60-sec
        n1m = 60-sec

        print(f"\n  ULTRA AI v11  |  {now.strftime('%H:%M:%S')}  "
              f"|  30s:{n30:>2}s  |  1m:{n1m:>2}s  |  {self.db.status()}")
        print(f"  CYCLE DETECTION + LIVE SELF-LEARNING | NO SKIP\n")

        for game in ['30s','1m']:
            g   = self.g[game]
            dkt = self.dk30 if game=='30s' else self.dk1m
            fxt = self.fx30 if game=='30s' else self.fx1m

            print(f"  {'═'*70}")
            print(f"  {'30 SECOND' if game=='30s' else '1 MINUTE'} MARKET"
                  f"  |  {self.dark.total} periods loaded")

            # Last 30 values for visual cycle check
            h30 = g['hist'][:30]
            if h30:
                sz = ''.join(['B' if x['size']=='big' else 'S' for x in h30])
                cl = ''.join(['G' if x['color']=='green'
                               else ('R' if x['color']=='red' else 'V')
                               for x in h30])
                print(f"  SIZE  (newest→) : {sz}")
                print(f"  COLOR (newest→) : {cl}")

            # Cycle detector state
            cv,cc,cstr,cscore = self.dark.cycle.predict() if True else (None,0,'',0)
            sv,sc,cstr2,cscore2 = self.fox.cycle.predict()
            print(f"  COLOR cycle: [{cstr}] score={cscore:.0f}%   "
                  f"SIZE cycle: [{cstr2}] score={cscore2:.0f}%")

            # Accuracy
            print(f"  DARK accuracy: {self.dark.scorer.display()}")
            print(f"  FOX  accuracy: {self.fox.scorer.display()}")

            print(f"  {'─'*70}")

            # DARK block
            fl = "  [FLIP]" if self.dark.flip_active else ""
            print(f"  DARK [COLOR]  WR:{dkt.wr()}%  "
                  f"W:{dkt.wins} L:{dkt.losses}  "
                  f"Wstr:{dkt.wstrk} Lstr:{dkt.lstrk}{fl}")
            self._pred_line(dkt.last, g['dkp'], g['dkt'], 'color')

            print()

            # FOX block
            fl2 = "  [FLIP]" if self.fox.flip_active else ""
            print(f"  FOX  [SIZE]   WR:{fxt.wr()}%  "
                  f"W:{fxt.wins} L:{fxt.losses}  "
                  f"Wstr:{fxt.wstrk} Lstr:{fxt.lstrk}{fl2}")
            self._pred_line(fxt.last, g['fxp'], g['fxt'], 'size')

            print()

    def _pred_line(self, last, p, target, vtype):
        # Last result line
        if last:
            w   = "WIN " if last['is_win'] else "LOSS"
            if vtype == 'color':
                a = f"{last['actual_c'].upper()}({last['actual_n']})"
            else:
                a = f"{last['actual_s'].upper()}({last['actual_n']})"
            print(f"  | Last [{w}]: pred={last['pred'].upper():<6} "
                  f"actual={a:<12} method={last['method']}")
        else:
            print(f"  | Last: waiting...")

        # Next prediction
        if not p:
            print(f"  | Next: analyzing...")
            return

        val  = p['value'].upper()
        conf = p['conf']
        bar  = '█'*int(conf/100*25) + '░'*(25-int(conf/100*25))

        print(f"  | Next period {target}:")
        print(f"  |   [{bar}] {conf}%  →  {val}")
        print(f"  |   {p['reason']}")

    # ── loops ─────────────────────────────────────────────────────────
    def loop(self, url, game, offset=0):
        time.sleep(offset)
        while True:
            try:
                d = self.fetch(url)
                if d: self.process(d, game)
                sec  = datetime.now().second
                if game == '30s':
                    wait = (30-sec+1) if sec<30 else (60-sec+1)
                else:
                    wait = 60-sec+1
                time.sleep(max(2, wait))
            except: time.sleep(3)

    def clock(self):
        time.sleep(8)
        while True:
            try: self.draw()
            except: pass
            time.sleep(1)

    def start(self):
        os.system('cls' if os.name=='nt' else 'clear')
        print("""
  ULTRA AI v11  —  STARTING
  ─────────────────────────────────────────────────────────
  Cycle detection: learns BBSBBSSS / RRGGRRRG patterns live
  After every result: cycle re-scored, weights updated
  Flip mode: 3 straight losses → prediction inverts
  No skip: every period gets a prediction
  ─────────────────────────────────────────────────────────
        """)
        time.sleep(2)
        threading.Thread(target=self.loop,  args=(self.URL30,'30s',0), daemon=True).start()
        threading.Thread(target=self.loop,  args=(self.URL1M,'1m', 1), daemon=True).start()
        threading.Thread(target=self.clock, daemon=True).start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopped.\n")

if __name__ == "__main__":
    Server().start()
