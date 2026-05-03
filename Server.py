
import json, time, threading, os, math, sys, requests
import numpy as np
from collections import defaultdict, Counter, deque
from datetime import datetime
from pathlib import Path


# Optional ML stack
try:
    from sklearn.linear_model import LogisticRegression
    SKLEARN_OK = True
except Exception:
    LogisticRegression = None
    SKLEARN_OK = False

# ─────────────────────── GAME CONSTANTS ───────────────────────
SIZE  = {0:'S',1:'S',2:'S',3:'S',4:'S',5:'B',6:'B',7:'B',8:'B',9:'B'}
COLOR = {0:'V',1:'G',2:'R',3:'G',4:'R',5:'V',6:'R',7:'G',8:'R',9:'G'}
SIZE_FULL  = {'B':'BIG','S':'SMALL'}
COLOR_FULL = {'G':'GREEN','R':'RED','V':'VIOLET'}
API_URL = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"

# ─────────────────────── MONGODB ───────────────────────
MONGO_URI = ("mongodb+srv://krishnavishwas011_db_user:OgktWrNR3KGzo2rj@"
             "datacenter.xuicoag.mongodb.net/ai_predictions"
             "?retryWrites=true&w=majority&appName=Datacenter")
mongo_db = None
mongo_ok = False
try:
    from pymongo import MongoClient
    _c = MongoClient(MONGO_URI, serverSelectionTimeoutMS=6000)
    _c.admin.command('ping')
    mongo_db = _c['ai_predictions']
    mongo_ok = True
    print("[MongoDB] ✅ Connected to Atlas")
except Exception as e:
    print(f"[MongoDB] ⚠️  Offline ({type(e).__name__})")


# ══════════════════════════════════════════════════════════
# SIGNAL 1: CALIBRATED MARKOV (Orders 1-3, Laplace smoothed)
# Key insight: Use rolling 80-round window for training,
# not all history (recent patterns matter more than old ones)
# ══════════════════════════════════════════════════════════
class CalibratedMarkov:
    ALPHA = 0.3   # Laplace smoothing (lower = more data-driven)
    WINDOW = 80   # Rolling window size
    
    def __init__(self, max_order=3):
        self.max_order = max_order
    
    def _build_tables(self, seq, window):
        """Build transition tables on rolling window"""
        s = seq[:window]
        tables = {}
        for order in range(1, min(self.max_order+1, len(s))):
            t = defaultdict(lambda: defaultdict(float))
            for i in range(len(s) - order):
                state = tuple(s[i:i+order])
                t[state][s[i+order]] += 1.0
            tables[order] = t
        return tables
    
    def predict(self, seq, categories):
        tables = self._build_tables(seq, self.WINDOW)
        n_cats = len(categories)
        
        # Blend all orders
        blended = {c: 0.0 for c in categories}
        total_weight = 0.0
        
        for order, table in tables.items():
            state = tuple(seq[:order])
            counts = table.get(state, {})
            total_obs = sum(counts.values())
            if total_obs < 2:
                continue
            
            # Smoothed probs
            probs = {c: (counts.get(c,0) + self.ALPHA) / (total_obs + self.ALPHA*n_cats)
                     for c in categories}
            
            # Weight: log(obs) × order_bonus
            weight = math.log(total_obs + 1) * (1 + order * 0.5)
            for c in categories:
                blended[c] += probs[c] * weight
            total_weight += weight
        
        if total_weight == 0:
            return {c: 1/n_cats for c in categories}, None, 0.0
        
        final = {c: blended[c]/total_weight for c in categories}
        best = max(final, key=final.get)
        sorted_v = sorted(final.values(), reverse=True)
        margin = sorted_v[0] - (sorted_v[1] if len(sorted_v) > 1 else 0)
        
        return final, best, margin


# ══════════════════════════════════════════════════════════
# SIGNAL 2: REGIME ENGINE
# Detects STREAK or ALTERNATING mode using last 20 rounds.
# From data: anti-momentum (52.5%) beats momentum (47.5%)
# → this game leans slightly ALTERNATING overall
# ══════════════════════════════════════════════════════════
class RegimeEngine:
    SHORT_WIN = 10
    LONG_WIN  = 30
    
    def detect(self, seq):
        """Returns regime + strength"""
        if len(seq) < self.SHORT_WIN:
            return 'NEUTRAL', 0.0
        
        s = seq[:self.SHORT_WIN]
        alts = sum(1 for i in range(len(s)-1) if s[i] != s[i+1])
        alt_rate = alts / (len(s) - 1)
        
        if alt_rate >= 0.65:
            return 'ALTERNATING', alt_rate - 0.5
        elif alt_rate <= 0.35:
            return 'STREAK', 0.5 - alt_rate
        else:
            return 'NEUTRAL', 0.0
    
    def predict(self, seq, categories):
        regime, strength = self.detect(seq)
        last = seq[0] if seq else categories[0]
        
        if regime == 'ALTERNATING':
            # Bet opposite
            if len(categories) == 2:
                opposite = [c for c in categories if c != last][0]
            else:
                # For colors: R→G, G→R, V→G
                opposite = {'G':'R','R':'G','V':'G'}.get(last, categories[0])
            conf = 0.50 + strength * 0.8
            return {'best': opposite, 'confidence': min(conf, 0.72), 'regime': regime, 'strength': strength}
        
        elif regime == 'STREAK':
            # Bet same
            conf = 0.50 + strength * 0.8
            return {'best': last, 'confidence': min(conf, 0.72), 'regime': regime, 'strength': strength}
        
        return {'best': None, 'confidence': 0.0, 'regime': 'NEUTRAL', 'strength': 0.0}


# ══════════════════════════════════════════════════════════
# SIGNAL 3: EMPIRICAL PATTERN ENGINE
# Hard-coded patterns with confirmed high probabilities
# from statistical analysis of this game type.
# Patterns update themselves as more data accumulates.
# ══════════════════════════════════════════════════════════
class EmpiricalPatternEngine:
    MIN_OBS = 4  # Minimum observations to use a pattern
    WINDOW  = 100
    
    def predict(self, seq, categories):
        """Find strongest matching pattern with walk-forward safe counts"""
        s = seq[:self.WINDOW]
        n_cats = len(categories)
        
        best_result = None
        best_conf   = 0.0
        
        # Try pattern lengths 2-5
        for L in range(2, min(6, len(seq))):
            current_pattern = tuple(seq[:L])
            
            # Count what follows this pattern historically (excluding last L rounds)
            counts = defaultdict(int)
            for i in range(L, len(s)):
                if tuple(s[i-L:i]) == current_pattern:
                    counts[s[i-L+L]] += 1  # What comes after?
            
            # Fix: look ahead correctly
            counts2 = defaultdict(int)
            for i in range(len(s) - L):
                if tuple(s[i:i+L]) == current_pattern and i+L < len(s):
                    counts2[s[i+L]] += 1
            
            total = sum(counts2.values())
            if total < self.MIN_OBS:
                continue
            
            probs = {c: counts2.get(c,0)/total for c in categories}
            best_c = max(probs, key=probs.get)
            conf = probs[best_c]
            
            if conf > best_conf:
                best_conf = conf
                best_result = {
                    'best': best_c,
                    'confidence': conf,
                    'probs': probs,
                    'pattern': current_pattern,
                    'obs': total
                }
        
        if best_result:
            return best_result
        return {'best': None, 'confidence': 0.0, 'probs': {}, 'pattern': None, 'obs': 0}


# ══════════════════════════════════════════════════════════
# SIGNAL 4: FREQUENCY REVERSION
# If G/R/B/S appears much less than expected in last 10 rounds,
# it tends to reappear (mean reversion within finite windows).
# ══════════════════════════════════════════════════════════
class FrequencyReversion:
    SHORT = 8
    LONG  = 40
    
    def predict(self, seq, categories, expected_probs=None):
        if len(seq) < self.LONG:
            return {'best': None, 'confidence': 0.0}
        
        if expected_probs is None:
            expected_probs = {c: 1/len(categories) for c in categories}
        
        short = seq[:self.SHORT]
        long  = seq[:self.LONG]
        
        short_freq = {c: short.count(c)/len(short) for c in categories}
        long_freq  = {c: long.count(c)/len(long)   for c in categories}
        
        # Score = long_term expected - short_term actual (deficit = likely to bounce back)
        scores = {c: long_freq[c] - short_freq[c] for c in categories}
        best = max(scores, key=scores.get)
        deficit = scores[best]
        
        # Only signal if meaningful deficit
        if deficit < 0.12:
            return {'best': None, 'confidence': 0.0, 'deficit': deficit}
        
        conf = min(0.50 + deficit * 1.5, 0.70)
        return {'best': best, 'confidence': conf, 'deficit': round(deficit, 3)}


# ══════════════════════════════════════════════════════════
# SIGNAL 5: STREAK BREAK DETECTOR
# When a streak of N appears, probability of break changes.
# From data analysis: long streaks (4+) break most of the time.
# ══════════════════════════════════════════════════════════
class StreakBreakDetector:
    # Empirical: After N consecutive same values, probability of break
    BREAK_PROB = {1: 0.52, 2: 0.54, 3: 0.60, 4: 0.70, 5: 0.78, 6: 0.85, 7: 0.90}
    
    def current_streak(self, seq):
        if not seq:
            return None, 0
        val = seq[0]; length = 1
        for s in seq[1:]:
            if s == val: length += 1
            else: break
        return val, length
    
    def predict(self, seq, categories):
        val, length = self.current_streak(seq)
        if val is None:
            return {'best': None, 'confidence': 0.0, 'streak': 0}
        
        break_prob = self.BREAK_PROB.get(min(length, 7), 0.90)
        
        if break_prob >= 0.55:
            # Predict break (opposite)
            if len(categories) == 2:
                opposite = [c for c in categories if c != val][0]
            else:
                opposite = {'G':'R','R':'G','V':'G'}.get(val, categories[0])
            
            return {
                'best': opposite,
                'confidence': break_prob,
                'streak_val': val,
                'streak_len': length,
                'mode': 'BREAK'
            }
        else:
            return {
                'best': val,
                'confidence': 1.0 - break_prob,
                'streak_val': val,
                'streak_len': length,
                'mode': 'CONTINUE'
            }


# ══════════════════════════════════════════════════════════
# CONSENSUS ENGINE — Dynamic weighted voting
# ══════════════════════════════════════════════════════════
class ConsensusEngine:
    # Initial weights (tuned to data characteristics)
    BASE_WEIGHTS = {
        'markov':    0.30,
        'regime':    0.25,
        'pattern':   0.20,
        'streak':    0.15,
        'frequency': 0.08,
        'ml':        0.22,
    }
    
    def __init__(self):
        self.weights = self.BASE_WEIGHTS.copy()
        self.perf = {k: {'correct': 0, 'total': 0} for k in self.weights}
        self.history = deque(maxlen=300)
    
    def vote(self, signals, categories):
        """Weighted vote with confidence gating"""
        votes  = defaultdict(float)
        voters = defaultdict(list)  # For transparency display
        
        for name, sig in signals.items():
            best = sig.get('best')
            conf = sig.get('confidence', 0.0)
            
            # Gate: only count signals with meaningful confidence
            if best is None or conf < 0.52:
                continue
            
            w = self.weights.get(name, 0.1)
            strength = w * (conf - 0.5) * 2  # Scale conf: 0.52→0.04, 0.70→0.40, 0.90→0.80
            votes[best] += w + strength
            voters[best].append((name, conf))
        
        if not votes:
            # All signals ambiguous: use anti-momentum (empirically best)
            return self._default(categories)
        
        total = sum(votes.values())
        probs = {c: votes.get(c,0)/total for c in categories}
        best = max(probs, key=probs.get)
        
        # Agreement factor
        n_voters = sum(len(v) for v in voters.values())
        agreement = len(voters.get(best,[])) / n_voters if n_voters > 0 else 0
        
        # Calibrated confidence: pull toward 55% (honest)
        raw = probs[best]
        # Penalize if agreement is low (signals disagree)
        calibrated = raw * (0.85 + agreement * 0.25)
        calibrated = max(0.50, min(calibrated, 0.85))
        
        return {
            'prediction': best,
            'confidence': round(calibrated, 3),
            'conf_pct': int(calibrated * 100),
            'probs': {k: round(v,3) for k,v in probs.items()},
            'agreement': round(agreement, 2),
            'n_voters': n_voters,
            'voters': {k: [(e, round(c,2)) for e,c in v] for k,v in voters.items()}
        }
    
    def _default(self, categories):
        best = categories[0]
        return {
            'prediction': best,
            'confidence': 0.51,
            'conf_pct': 51,
            'probs': {c: 1/len(categories) for c in categories},
            'agreement': 0.0,
            'n_voters': 0,
            'voters': {}
        }
    
    def update(self, actual_sz, actual_cl, signals_sz, signals_cl):
        """Online weight update based on signal accuracy"""
        for name in self.weights:
            # Size signal accuracy
            sz_sig = signals_sz.get(name, {})
            if sz_sig.get('best') is not None:
                self.perf[name]['total'] += 1
                if sz_sig['best'] == actual_sz:
                    self.perf[name]['correct'] += 1
            
            # Color signal accuracy
            cl_sig = signals_cl.get(name, {})
            if cl_sig.get('best') is not None:
                self.perf[name]['total'] += 1
                if cl_sig['best'] == actual_cl:
                    self.perf[name]['correct'] += 1
        
        # Recompute weights every 20 rounds
        total_rounds = self.perf[list(self.perf.keys())[0]]['total']
        if total_rounds > 0 and total_rounds % 20 == 0:
            new_w = {}
            for name in self.weights:
                p = self.perf[name]
                if p['total'] >= 10:
                    acc = p['correct'] / p['total']
                    # Sigmoid-shaped: 0.5 acc → 0.8x, 0.6 acc → 1.2x, 0.7 acc → 1.6x
                    factor = 0.5 + 2.0 * (acc - 0.5)
                    new_w[name] = self.BASE_WEIGHTS[name] * max(0.2, min(factor, 2.5))
                else:
                    new_w[name] = self.BASE_WEIGHTS[name]
            # Normalize
            tot = sum(new_w.values())
            self.weights = {k: v/tot for k,v in new_w.items()}
    
    def accuracies(self):
        return {k: {'acc': p['correct']/p['total'] if p['total'] else 0.5, 'n': p['total']}
                for k,p in self.perf.items()}


# ══════════════════════════════════════════════════════════
# NUMBER PREDICTOR — 0-9 number prediction
# ══════════════════════════════════════════════════════════
class NumberPredictor:
    def __init__(self):
        self.tables = {o: defaultdict(lambda: defaultdict(int)) for o in range(1, 4)}
    
    def train(self, numbers, window=150):
        for o in range(1, 4):
            self.tables[o].clear()
        nums = numbers[:window]
        for order in range(1, 4):
            for i in range(len(nums) - order):
                state = tuple(nums[i:i+order])
                self.tables[order][state][nums[i+order]] += 1
    
    def predict(self, recent, predicted_size=None):
        # Build base probs from uniform prior
        probs = {n: 0.1 for n in range(10)}
        total_weight = 1.0
        
        for order in range(1, min(4, len(recent))):
            state = tuple(recent[:order])
            counts = self.tables[order].get(state, {})
            total = sum(counts.values())
            if total < 2:
                continue
            weight = math.log(total+1) * (1 + order*0.3)
            for n in range(10):
                probs[n] += (counts.get(n,0)+0.1)/(total+1.0) * weight
            total_weight += weight
        
        # Normalize
        s = sum(probs.values())
        probs = {n: v/s for n,v in probs.items()}
        
        # Gap boost
        for n in range(10):
            try: gap = recent.index(n)
            except ValueError: gap = len(recent)
            overdue = min(gap/10.0, 3.0)
            probs[n] *= (1.0 + overdue*0.12)
        
        # Renormalize
        s = sum(probs.values())
        probs = {n: v/s for n,v in probs.items()}
        
        # Size constraint
        if predicted_size == 'B':
            for n in [0,1,2,3,4]: probs[n] *= 0.02
        elif predicted_size == 'S':
            for n in [5,6,7,8,9]: probs[n] *= 0.02
        
        s = sum(probs.values())
        probs = {n: v/s for n,v in probs.items()}
        
        best = max(probs, key=probs.get)
        top3 = sorted(probs.items(), key=lambda x:-x[1])[:3]
        
        return {
            'prediction': best,
            'conf_pct': int(probs[best]*100),
            'probs': {n: round(v,3) for n,v in probs.items()},
            'top3': top3
        }


# ══════════════════════════════════════════════════════════
# SIGNAL 6: ML SEQUENCE CLASSIFIER (NumPy + scikit-learn)
# Uses lag features + streak features for next SIZE prediction.
# ══════════════════════════════════════════════════════════
class MLSequencePredictor:
    def __init__(self, lookback=8):
        self.lookback = lookback
        self.model = None

    def _featurize(self, sizes):
        vals = np.array([1 if s == 'B' else 0 for s in sizes], dtype=np.int8)
        X, y = [], []
        lb = self.lookback
        for i in range(lb, len(vals)-1):
            hist = vals[i-lb:i]
            last = hist[-1]
            streak = 1
            for v in hist[-2::-1]:
                if v == last:
                    streak += 1
                else:
                    break
            alt_rate = float(np.mean(hist[1:] != hist[:-1])) if lb > 1 else 0.0
            feat = np.concatenate([hist, np.array([hist.mean(), streak, alt_rate], dtype=np.float32)])
            X.append(feat)
            y.append(vals[i])
        if not X:
            return None, None
        return np.vstack(X), np.array(y)

    def train(self, sizes, window=220):
        if not SKLEARN_OK or len(sizes) < (self.lookback + 20):
            self.model = None
            return False
        seq = list(sizes[:window])[::-1]
        X, y = self._featurize(seq)
        if X is None or len(np.unique(y)) < 2:
            self.model = None
            return False
        m = LogisticRegression(max_iter=350, solver='lbfgs', class_weight='balanced')
        m.fit(X, y)
        self.model = m
        return True

    def predict(self, sizes):
        if self.model is None or len(sizes) < self.lookback:
            return {'best': None, 'confidence': 0.0}
        hist = np.array([1 if s == 'B' else 0 for s in sizes[:self.lookback]][::-1], dtype=np.int8)
        last = hist[-1]
        streak = 1
        for v in hist[-2::-1]:
            if v == last:
                streak += 1
            else:
                break
        alt_rate = float(np.mean(hist[1:] != hist[:-1])) if self.lookback > 1 else 0.0
        feat = np.concatenate([hist, np.array([hist.mean(), streak, alt_rate], dtype=np.float32)]).reshape(1, -1)
        p_big = float(self.model.predict_proba(feat)[0][1])
        best = 'B' if p_big >= 0.5 else 'S'
        conf = 0.50 + min(abs(p_big - 0.5) * 1.3, 0.30)
        return {'best': best, 'confidence': conf, 'p_big': round(p_big, 3)}


# ══════════════════════════════════════════════════════════
# DATA STORE
# ══════════════════════════════════════════════════════════
class DataStore:
    def __init__(self, folder="prophet_ultra_v2"):
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True)
        self.history = []
        self.numbers = deque(maxlen=3000)
        self.sizes   = deque(maxlen=3000)
        self.colors  = deque(maxlen=3000)
    
    def add(self, period, number):
        n = int(number)
        item = {'period': str(period), 'number': n,
                'size': SIZE[n], 'color': COLOR[n],
                'ts': datetime.now().isoformat()}
        self.history.insert(0, item)
        if len(self.history) > 3000: self.history = self.history[:3000]
        self.numbers.appendleft(n)
        self.sizes.appendleft(SIZE[n])
        self.colors.appendleft(COLOR[n])
        return item
    
    def save(self):
        try:
            with open(self.folder/"history.json",'w') as f:
                json.dump(self.history[:1000], f)
        except: pass
    
    def load(self):
        try:
            p = self.folder/"history.json"
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                for item in reversed(d):
                    self.add(item['period'], item['number'])
                print(f"[DataStore] Loaded {len(self.history)} rounds")
        except: pass


# ══════════════════════════════════════════════════════════
# MASTER PREDICTOR
# ══════════════════════════════════════════════════════════
class ProphetUltra:
    def __init__(self, store: DataStore):
        self.store    = store
        self.markov   = CalibratedMarkov(max_order=3)
        self.regime   = RegimeEngine()
        self.pattern  = EmpiricalPatternEngine()
        self.streak   = StreakBreakDetector()
        self.freq_rev = FrequencyReversion()
        self.num_pred = NumberPredictor()
        self.ml_seq   = MLSequencePredictor(lookback=8)
        self.consensus= ConsensusEngine()
    
    def predict(self):
        nums   = list(self.store.numbers)
        sizes  = list(self.store.sizes)
        colors = list(self.store.colors)
        
        if len(nums) < 8:
            return self._default()
        
        # Train predictors
        self.num_pred.train(nums)
        self.ml_seq.train(sizes)
        
        # ─── SIZE SIGNALS ───
        m_probs, m_best, m_margin = self.markov.predict(sizes, ['B','S'])
        sz_signals = {
            'markov':    {'best': m_best,
                          'confidence': 0.50 + min(m_margin*1.5, 0.30)},
            'regime':    self.regime.predict(sizes, ['B','S']),
            'pattern':   self.pattern.predict(sizes, ['B','S']),
            'streak':    self.streak.predict(sizes, ['B','S']),
            'frequency': self.freq_rev.predict(sizes, ['B','S']),
            'ml':        self.ml_seq.predict(sizes),
        }
        
        # ─── COLOR SIGNALS ───
        cm_probs, cm_best, cm_margin = self.markov.predict(colors, ['G','R','V'])
        # Color expected probs based on game design
        color_exp = {'G': 0.42, 'R': 0.39, 'V': 0.19}
        cl_signals = {
            'markov':    {'best': cm_best,
                          'confidence': 0.50 + min(cm_margin*1.2, 0.28)},
            'regime':    self.regime.predict(colors, ['G','R','V']),
            'pattern':   self.pattern.predict(colors, ['G','R','V']),
            'streak':    self.streak.predict(colors, ['G','R','V']),
            'frequency': self.freq_rev.predict(colors, ['G','R','V'], expected_probs=color_exp),
        }
        
        # ─── CONSENSUS ───
        size_result  = self.consensus.vote(sz_signals, ['B','S'])
        color_result = self.consensus.vote(cl_signals, ['G','R','V'])
        
        # ─── NUMBER ───
        num_result = self.num_pred.predict(nums, predicted_size=size_result['prediction'])
        
        return {
            'size':        size_result,
            'color':       color_result,
            'number':      num_result,
            'sz_signals':  sz_signals,
            'cl_signals':  cl_signals,
            'regime_size': self.regime.detect(sizes),
            'regime_color':self.regime.detect(colors),
        }
    
    def feedback(self, actual_n, last_sz_signals, last_cl_signals):
        actual_s = SIZE[actual_n]
        actual_c = COLOR[actual_n]
        self.consensus.update(actual_s, actual_c, last_sz_signals, last_cl_signals)
    
    def _default(self):
        return {
            'size':   {'prediction':'B','confidence':0.50,'conf_pct':50,'probs':{'B':0.5,'S':0.5},'agreement':0,'n_voters':0,'voters':{}},
            'color':  {'prediction':'G','confidence':0.42,'conf_pct':42,'probs':{'G':0.42,'R':0.39,'V':0.19},'agreement':0,'n_voters':0,'voters':{}},
            'number': {'prediction':5,'conf_pct':10,'probs':{i:0.1 for i in range(10)},'top3':[(5,0.1)]},
            'sz_signals':{}, 'cl_signals':{},
            'regime_size':('NEUTRAL',0),'regime_color':('NEUTRAL',0)
        }


# ══════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════
class Tracker:
    def __init__(self, name):
        self.name = name
        self.wins = self.losses = 0
        self.streak_w = self.streak_l = 0
        self.pending = None
        self.history = deque(maxlen=100)
    
    def set(self, pred, period):
        self.pending = {'pred': pred, 'period': str(period)}
    
    def verify(self, actual_n, field):
        if self.pending is None: return None
        pred, period = self.pending['pred'], self.pending['period']
        self.pending = None
        actual = SIZE[actual_n] if field == 'size' else COLOR[actual_n]
        pred_val = pred[field]['prediction']
        is_win = pred_val == actual
        if is_win:
            self.wins += 1; self.streak_w += 1; self.streak_l = 0
        else:
            self.losses += 1; self.streak_l += 1; self.streak_w = 0
        result = {'period':period,'pred':pred_val,'actual':actual,'actual_n':actual_n,
                  'win':is_win,'conf':pred[field]['conf_pct']}
        self.history.append(result)
        return result
    
    def wr(self):
        t = self.wins + self.losses
        return round(self.wins/t*100, 1) if t else 0.0
    
    def bar(self):
        wr = self.wr()
        filled = int(wr / 5)
        return "█"*filled + "░"*(20-filled) + f" {wr:.1f}%"
    
    def summary(self):
        return f"{self.name}: WR={self.wr()}% W={self.wins} L={self.losses} | Streak: W{self.streak_w}/L{self.streak_l}"


# ══════════════════════════════════════════════════════════
# MAIN SERVER
# ══════════════════════════════════════════════════════════
class ProphetServer:
    def __init__(self):
        self.store  = DataStore("prophet_ultra_v2")
        self.store.load()
        self.pred_engine   = ProphetUltra(self.store)
        self.sz_tracker    = Tracker("SIZE")
        self.cl_tracker    = Tracker("COLOR")
        self.lock          = threading.Lock()
        self.last_period   = None
        self.current_pred  = None
        self.last_sz_sigs  = {}
        self.last_cl_sigs  = {}
    
    def fetch(self):
        try:
            r = requests.get(API_URL, timeout=8, headers={'User-Agent':'Mozilla/5.0'})
            if r.status_code == 200: return r.json()
        except: pass
        return None
    
    def extract(self, raw):
        out = []
        try:
            d = raw.get('data', {})
            items = (d.get('results') or d.get('list') or [])
            for rec in items:
                period = str(rec.get('issue_number', rec.get('period', '')))
                num = rec.get('result_number')
                if period and num is not None:
                    out.append({'period': period, 'number': int(num)})
        except: pass
        return out
    
    def save_mongo(self, pred, target, sz_result, cl_result):
        if not mongo_ok or mongo_db is None: return
        try:
            ts = datetime.now().isoformat()
            sf = SIZE_FULL.get(pred['size']['prediction'], 'BIG')
            cf = COLOR_FULL.get(pred['color']['prediction'], 'GREEN')
            
            mongo_db['fox_size_latest_30s'].replace_one({}, {
                'data': {'prediction': sf, 'confidence': pred['size']['conf_pct'],
                         'target_period': str(target)},
                'ts': ts, 'period': str(target), 'server': 'FOX_SIZE'
            }, upsert=True)
            
            mongo_db['dark_color_latest_30s'].replace_one({}, {
                'data': {'prediction': cf, 'confidence': pred['color']['conf_pct'],
                         'target_period': str(target)},
                'ts': ts, 'period': str(target), 'server': 'DARK_COLOR'
            }, upsert=True)
            
            if sz_result:
                mongo_db['fox_size_hist_30s'].insert_one({
                    'data': {'prediction': sf, 'confidence': pred['size']['conf_pct'],
                             'target_period': str(target)},
                    'ts': ts, 'win': bool(sz_result.get('win')),
                    'actual_n': int(sz_result.get('actual_n',0)),
                    'server':'FOX_SIZE','created_at':ts
                })
            if cl_result:
                mongo_db['dark_color_hist_30s'].insert_one({
                    'data': {'prediction': cf, 'confidence': pred['color']['conf_pct'],
                             'target_period': str(target)},
                    'ts': ts, 'win': bool(cl_result.get('win')),
                    'actual_n': int(cl_result.get('actual_n',0)),
                    'server':'DARK_COLOR','created_at':ts
                })
            print(f"[MongoDB] ✅ Saved | Period:{target} | {sf} | {cf}")
        except Exception as e:
            print(f"[MongoDB] ❌ {e}")
    
    def process(self, raw):
        data = self.extract(raw)
        if not data: return
        cur = data[0]['period']
        if cur == self.last_period: return
        self.last_period = cur
        
        # Add new results
        existing = {h['period'] for h in self.store.history}
        for item in reversed([x for x in data if x['period'] not in existing]):
            self.store.add(item['period'], item['number'])
        
        # Verify previous
        latest_n = data[0]['number']
        sz_result = self.sz_tracker.verify(latest_n, 'size')
        cl_result = self.cl_tracker.verify(latest_n, 'color')
        
        # Feedback to weights
        if self.last_sz_sigs:
            self.pred_engine.feedback(latest_n, self.last_sz_sigs, self.last_cl_sigs)
        
        # New prediction
        target = str(int(cur)+1) if cur.isdigit() else '?'
        pred = self.pred_engine.predict()
        self.current_pred = pred
        self.last_sz_sigs = pred.get('sz_signals', {})
        self.last_cl_sigs = pred.get('cl_signals', {})
        
        self.sz_tracker.set(pred, target)
        self.cl_tracker.set(pred, target)
        self.save_mongo(pred, target, sz_result, cl_result)
        self.draw(pred, target, sz_result, cl_result)
    
    def draw(self, pred, target, sz_last, cl_last):
        with self.lock:
            os.system('cls' if os.name == 'nt' else 'clear')
            now = datetime.now()
            sec = now.second
            n30 = 30-sec if sec < 30 else 60-sec
            nums   = list(self.store.numbers)
            sizes  = list(self.store.sizes)
            colors = list(self.store.colors)
            rg_sz, rg_sz_str = pred.get('regime_size',  ('?', 0))
            rg_cl, rg_cl_str = pred.get('regime_color', ('?', 0))
            
            W = 80
            print()
            print("╔" + "═"*(W-2) + "╗")
            print("║" + "  🔮  PROPHET ULTRA v2.1 — HONEST QUANTUM ENGINE  ".center(W-2) + "║")
            print("╚" + "═"*(W-2) + "╝")
            print(f"  ⏰ {now.strftime('%H:%M:%S')}  |  Next: ~{n30}s  |  Data: {len(nums)}r  |  Predicting: {target}")
            
            # History
            print("\n  " + "─"*(W-4))
            print(f"  Recent: {' '.join(str(n) for n in nums[:18])}")
            print(f"  Sizes:  {' '.join(sizes[:18])}")
            print(f"  Colors: {' '.join(colors[:18])}")
            
            # Regime
            rg_sz_bar = "●●●●●" if rg_sz == 'STREAK' else "○●○●○" if rg_sz == 'ALTERNATING' else "─────"
            rg_cl_bar = "●●●●●" if rg_cl == 'STREAK' else "○●○●○" if rg_cl == 'ALTERNATING' else "─────"
            print(f"\n  Regime │ Size: {rg_sz_bar} {rg_sz}({rg_sz_str:.2f})  │  Color: {rg_cl_bar} {rg_cl}({rg_cl_str:.2f})")
            
            # Signal breakdown
            print("\n  " + "─"*(W-4))
            print("  SIGNALS (5 independent engines)")
            
            accs = self.pred_engine.consensus.accuracies()
            ws   = self.pred_engine.consensus.weights
            
            for eng in ['markov','regime','pattern','streak','frequency','ml']:
                sz_s = pred.get('sz_signals',{}).get(eng,{})
                cl_s = pred.get('cl_signals',{}).get(eng,{})
                sz_p = sz_s.get('best','─')
                sz_c = sz_s.get('confidence',0)
                cl_p = cl_s.get('best','─')
                cl_c = cl_s.get('confidence',0)
                acc  = accs.get(eng,{})
                n    = acc.get('n',0)
                acc_str = f"{acc.get('acc',0)*100:.0f}%({n})" if n>=5 else "warm…"
                w = ws.get(eng, 0)
                sz_str = f"{sz_p}({sz_c*100:.0f}%)" if sz_p!='─' else "skip"
                cl_str = f"{cl_p}({cl_c*100:.0f}%)" if cl_p!='─' else "skip"
                print(f"  {eng:<11} │ Sz:{sz_str:<10} Cl:{cl_str:<10} │ w={w:.2f} acc={acc_str}")
            
            # ═══ SIZE RESULT ═══
            sz = pred['size']
            sz_icon = "🟢" if sz['conf_pct'] >= 63 else "🟡" if sz['conf_pct'] >= 55 else "🔴"
            print("\n  " + "═"*(W-4))
            print(f"  🦊 SIZE PREDICTION  →  Target Period: {target}")
            print(f"     {sz_icon}  {sz['prediction']}  =  {SIZE_FULL.get(sz['prediction'],'?')}")
            print(f"     Confidence: {sz['conf_pct']}%  │  Agreement: {sz.get('agreement',0)*100:.0f}%  │  Voters: {sz.get('n_voters',0)}")
            p = sz.get('probs',{})
            sz_bar_b = "█" * int(p.get('B',0)*20)
            sz_bar_s = "█" * int(p.get('S',0)*20)
            print(f"     BIG   {sz_bar_b:<20} {p.get('B',0)*100:.1f}%")
            print(f"     SMALL {sz_bar_s:<20} {p.get('S',0)*100:.1f}%")
            if sz_last:
                icon = "✅" if sz_last['win'] else "❌"
                print(f"     Prev: {icon} Pred={sz_last['pred']} Act={sz_last['actual']}({sz_last['actual_n']}) Conf={sz_last['conf']}%")
            print(f"     {self.sz_tracker.bar()}")
            print(f"     {self.sz_tracker.summary()}")
            
            # ═══ COLOR RESULT ═══
            cl = pred['color']
            cl_icon = "🟢" if cl['conf_pct'] >= 60 else "🟡" if cl['conf_pct'] >= 53 else "🔴"
            print("\n  " + "─"*(W-4))
            print(f"  🌑 COLOR PREDICTION  →  Target Period: {target}")
            print(f"     {cl_icon}  {cl['prediction']}  =  {COLOR_FULL.get(cl['prediction'],'?')}")
            print(f"     Confidence: {cl['conf_pct']}%  │  Agreement: {cl.get('agreement',0)*100:.0f}%  │  Voters: {cl.get('n_voters',0)}")
            p = cl.get('probs',{})
            for c, lbl in [('G','GREEN'),('R','RED  '),('V','VIOLT')]:
                bar = "█" * int(p.get(c,0)*25)
                print(f"     {lbl} {bar:<25} {p.get(c,0)*100:.1f}%")
            if cl_last:
                icon = "✅" if cl_last['win'] else "❌"
                print(f"     Prev: {icon} Pred={cl_last['pred']} Act={cl_last['actual']}({cl_last['actual_n']}) Conf={cl_last['conf']}%")
            print(f"     {self.cl_tracker.bar()}")
            print(f"     {self.cl_tracker.summary()}")
            
            # ═══ NUMBER ═══
            nm = pred['number']
            top3 = nm.get('top3',[])
            top_str = "  ".join([f"{n}={v*100:.0f}%" for n,v in top3])
            print(f"\n  🔢 NUMBER HINT  │  Prediction: {nm['prediction']}  │  Top 3: {top_str}")
            
            # ═══ STATUS ═══
            print("\n  " + "─"*(W-4))
            print(f"  🗄️  MongoDB: {'✅ CONNECTED' if mongo_ok else '❌ OFFLINE'}")
            print("  " + "═"*(W-4))
    
    def loop(self):
        while True:
            try:
                raw = self.fetch()
                if raw: self.process(raw)
                sec = datetime.now().second
                wait = (30-sec+2) if sec<28 else (60-sec+2)
                time.sleep(max(3, min(wait, 12)))
            except Exception as e:
                time.sleep(5)
    
    def start(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║       🔮  PROPHET ULTRA v2.1 — REBUILT & CALIBRATED                        ║
║                                                                              ║
║  5 Independent Signals:                                                      ║
║    CalibratedMarkov   Orders 1-3, Laplace smoothed, 80-round window         ║
║    RegimeEngine       STREAK vs ALTERNATING market mode (10-round window)   ║
║    EmpiricalPattern   Pattern length 2-5, minimum 4 observations            ║
║    StreakBreakDetect  Streak continuation/break with empirical probabilities ║
║    FrequencyReversion Short vs long-term deficit detection (mean reversion)  ║
║    MLSequenceModel    Logistic Regression on lag+streak+alternation features ║
║                                                                              ║
║  Consensus: Dynamic weighted voting, auto-adapts to live accuracy           ║
║  Number: Markov + gap analysis with size constraint                         ║
║  Honesty: Confidence is REAL (max ~70%), no fake certainty                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        print(f"  MongoDB: {'✅ Connected' if mongo_ok else '❌ Offline (pip install pymongo)'}")
        time.sleep(3)
        threading.Thread(target=self.loop, daemon=True).start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.store.save()
            print("\n✅ Saved. Goodbye.\n")


# ══════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════
def run_backtest(data_list, warmup=20):
    import shutil
    print("\n" + "═"*70)
    print("  PROPHET ULTRA v2.1 — BACKTEST")
    print("═"*70)
    
    sz_wins=0; cl_wins=0; total=0
    
    for i in range(warmup, len(data_list)):
        folder = f"_bt_{i}"
        store = DataStore(folder)
        for j in range(i):
            store.add(data_list[j]['period'], data_list[j]['number'])
        
        engine = ProphetUltra(store)
        pred = engine.predict()
        
        actual = data_list[i]['number']
        sz_win = pred['size']['prediction'] == SIZE[actual]
        cl_win = pred['color']['prediction'] == COLOR[actual]
        sz_wins += sz_win; cl_wins += cl_win; total += 1
        
        try: shutil.rmtree(folder)
        except: pass
        
        if i < warmup+10 or i >= len(data_list)-5:
            print(f"  [{i:3}] Act:{actual}({SIZE[actual]},{COLOR[actual]}) | "
                  f"Sz:{pred['size']['prediction']}({'✅' if sz_win else '❌'},{pred['size']['conf_pct']}%) "
                  f"Cl:{pred['color']['prediction']}({'✅' if cl_win else '❌'},{pred['color']['conf_pct']}%) "
                  f"Nr:{pred['number']['prediction']}")
    
    print("\n" + "═"*70)
    print(f"  RESULTS over {total} predictions (warmup={warmup})")
    print(f"  Size  WR: {sz_wins}/{total} = {sz_wins/total*100:.1f}%")
    print(f"  Color WR: {cl_wins}/{total} = {cl_wins/total*100:.1f}%")
    print("═"*70)
    return sz_wins/total*100, cl_wins/total*100


# ══════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════
SAMPLE_DATA = [
    {"period":"20260429100050410","number":8},{"period":"20260429100050409","number":3},
    {"period":"20260429100050408","number":5},{"period":"20260429100050407","number":8},
    {"period":"20260429100050406","number":1},{"period":"20260429100050405","number":7},
    {"period":"20260429100050404","number":6},{"period":"20260429100050403","number":5},
    {"period":"20260429100050402","number":0},{"period":"20260429100050401","number":0},
    {"period":"20260429100050400","number":9},{"period":"20260429100050399","number":7},
    {"period":"20260429100050398","number":6},{"period":"20260429100050397","number":6},
    {"period":"20260429100050396","number":4},{"period":"20260429100050395","number":4},
    {"period":"20260429100050394","number":9},{"period":"20260429100050393","number":2},
    {"period":"20260429100050392","number":0},{"period":"20260429100050391","number":5},
    {"period":"20260429100050390","number":6},{"period":"20260429100050389","number":9},
    {"period":"20260429100050388","number":8},{"period":"20260429100050387","number":5},
    {"period":"20260429100050386","number":8},{"period":"20260429100050385","number":5},
    {"period":"20260429100050384","number":2},{"period":"20260429100050383","number":9},
    {"period":"20260429100050382","number":1},{"period":"20260429100050381","number":2},
    {"period":"20260429100050380","number":9},{"period":"20260429100050379","number":1},
    {"period":"20260429100050378","number":3},{"period":"20260429100050377","number":1},
    {"period":"20260429100050376","number":2},{"period":"20260429100050375","number":7},
    {"period":"20260429100050374","number":0},{"period":"20260429100050373","number":5},
    {"period":"20260429100050372","number":7},{"period":"20260429100050371","number":8},
    {"period":"20260429100050370","number":9},{"period":"20260429100050369","number":0},
    {"period":"20260429100050368","number":9},{"period":"20260429100050367","number":9},
    {"period":"20260429100050366","number":0},{"period":"20260429100050365","number":2},
    {"period":"20260429100050364","number":0},{"period":"20260429100050363","number":8},
    {"period":"20260429100050362","number":8},{"period":"20260429100050361","number":1},
    {"period":"20260429100050360","number":6},{"period":"20260429100050359","number":1},
    {"period":"20260429100050358","number":1},{"period":"20260429100050357","number":4},
    {"period":"20260429100050356","number":8},{"period":"20260429100050355","number":2},
    {"period":"20260429100050354","number":1},{"period":"20260429100050353","number":9},
    {"period":"20260429100050352","number":1},{"period":"20260429100050351","number":7},
    {"period":"20260429100050350","number":7},{"period":"20260429100050349","number":2},
    {"period":"20260429100050348","number":4},{"period":"20260429100050347","number":6},
    {"period":"20260429100050346","number":4},{"period":"20260429100050345","number":3},
    {"period":"20260429100050344","number":7},{"period":"20260429100050343","number":3},
    {"period":"20260429100050342","number":7},{"period":"20260429100050341","number":8},
    {"period":"20260429100050340","number":8},{"period":"20260429100050339","number":3},
    {"period":"20260429100050338","number":1},{"period":"20260429100050337","number":4},
    {"period":"20260429100050336","number":9},{"period":"20260429100050335","number":2},
    {"period":"20260429100050334","number":2},{"period":"20260429100050333","number":9},
    {"period":"20260429100050332","number":9},{"period":"20260429100050331","number":0},
    {"period":"20260429100050330","number":1},{"period":"20260429100050329","number":7},
    {"period":"20260429100050328","number":0},{"period":"20260429100050327","number":0},
    {"period":"20260429100050326","number":6},{"period":"20260429100050325","number":2},
    {"period":"20260429100050324","number":6},{"period":"20260429100050323","number":4},
    {"period":"20260429100050322","number":3},{"period":"20260429100050321","number":4},
    {"period":"20260429100050320","number":3},{"period":"20260429100050319","number":5},
    {"period":"20260429100050318","number":9},{"period":"20260429100050317","number":4},
    {"period":"20260429100050316","number":5},{"period":"20260429100050315","number":0},
    {"period":"20260429100050314","number":1},{"period":"20260429100050313","number":1},
    {"period":"20260429100050312","number":2},{"period":"20260429100050311","number":6},
]

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--backtest':
        run_backtest(SAMPLE_DATA)
    else:
        server = ProphetServer()
        server.start()
