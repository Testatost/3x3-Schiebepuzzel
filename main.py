import sys
import random
import heapq
from typing import List, Optional, Tuple, Dict

from PySide6.QtCore import (
    Qt, QRect, QEasingCurve, QPropertyAnimation, QParallelAnimationGroup, QTimer, QSize
)
from PySide6.QtGui import QFont, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QMessageBox, QFrame,
    QTextEdit, QSizePolicy, QFileDialog
)

# -----------------------------
# Puzzle-Logik
# -----------------------------

GOAL = [1, 2, 3,
        4, 5, 6,
        7, 8, 0]  # 0 = leer

GOAL_POS = {v: i for i, v in enumerate(GOAL)}

def inversions(state: List[int]) -> int:
    arr = [x for x in state if x != 0]
    inv = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inv += 1
    return inv

def is_solvable_3x3(state: List[int]) -> bool:
    # 3x3: lösbar <=> Inversionen gerade
    return inversions(state) % 2 == 0

def parse_state(text: str) -> Optional[List[int]]:
    """
    Erlaubt:
      "1 2 3 4 5 6 7 8 0"
      "1,2,3,4,5,6,7,8,0"
      "123456780"
    """
    t = text.strip()
    if not t:
        return None

    if all(ch.isdigit() for ch in t) and len(t) == 9:
        vals = [int(ch) for ch in t]
    else:
        for sep in [",", ";"]:
            t = t.replace(sep, " ")
        parts = [p for p in t.split() if p]
        if len(parts) != 9:
            return None
        try:
            vals = [int(p) for p in parts]
        except ValueError:
            return None

    if sorted(vals) != list(range(9)):
        return None
    return vals

def neighbors(index: int) -> List[int]:
    r, c = divmod(index, 3)
    out = []
    if r > 0: out.append(index - 3)
    if r < 2: out.append(index + 3)
    if c > 0: out.append(index - 1)
    if c < 2: out.append(index + 1)
    return out

def manhattan(state: Tuple[int, ...]) -> int:
    dist = 0
    for i, v in enumerate(state):
        if v == 0:
            continue
        gi = GOAL_POS[v]
        r1, c1 = divmod(i, 3)
        r2, c2 = divmod(gi, 3)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist

def astar_solve(start: List[int], max_expansions: int = 250000) -> Optional[List[int]]:
    """
    Gibt eine Liste der zu bewegenden Tile-Werte zurück (z.B. [8,5,2,...]) oder None.
    """
    start_t = tuple(start)
    goal_t = tuple(GOAL)
    if start_t == goal_t:
        return []

    open_heap = []
    g_cost: Dict[Tuple[int, ...], int] = {start_t: 0}
    parent: Dict[Tuple[int, ...], Tuple[Tuple[int, ...], int]] = {}  # child -> (parent_state, moved_tile)

    heapq.heappush(open_heap, (manhattan(start_t), 0, start_t))

    expansions = 0
    while open_heap:
        f, g, state = heapq.heappop(open_heap)
        if g != g_cost.get(state, 10**9):
            continue

        if state == goal_t:
            moves: List[int] = []
            cur = state
            while cur in parent:
                prev, moved = parent[cur]
                moves.append(moved)
                cur = prev
            moves.reverse()
            return moves

        expansions += 1
        if expansions > max_expansions:
            return None

        z = state.index(0)
        for nb in neighbors(z):
            moved_tile = state[nb]
            new_state = list(state)
            new_state[z], new_state[nb] = new_state[nb], new_state[z]
            new_t = tuple(new_state)

            new_g = g + 1
            if new_g < g_cost.get(new_t, 10**9):
                g_cost[new_t] = new_g
                parent[new_t] = (state, moved_tile)
                new_f = new_g + manhattan(new_t)
                heapq.heappush(open_heap, (new_f, new_g, new_t))

    return None


# -----------------------------
# GUI
# -----------------------------

class SlidingPuzzle(QWidget):
    TILE = 92
    GAP = 10
    PAD = 12
    ANIM_MS = 160
    PLAYBACK_GAP_MS = 40

    BASE_SIZE = QSize(420, 300)   # <-- Ausgangsfenstergröße

    BTN_W = 100
    BTN_H = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3x3 Schiebe-Puzzel")

        # Standardgröße 900x900 (und merken für "Log zu" Rücksprung)
        self.resize(self.BASE_SIZE)
        self._base_size = QSize(self.BASE_SIZE)

        self.state: List[int] = GOAL.copy()
        self.initial_state: List[int] = self.state.copy()

        self.tiles: Dict[int, QPushButton] = {}
        self._animating = False
        self._auto_playing = False
        self._pending_moves: List[int] = []
        self._anim_group: Optional[QParallelAnimationGroup] = None

        # Bildmodus
        self._image_mode = False
        self._base_image: Optional[QPixmap] = None
        self._tile_images: Dict[int, QPixmap] = {}

        self._build_ui()
        self._build_tiles()
        self._apply_tile_appearance()
        self._sync_tiles_to_state(animate=False)

        # Start: Log zu
        self.log_panel.setVisible(False)
        self.btn_log.setText("Log anzeigen")

        # damit Qt das Layout einmal final berechnet (hilft beim späteren Resizing)
        QTimer.singleShot(0, self._refresh_base_size)

    # ---------- UI ----------

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # Left/main column
        left = QVBoxLayout()
        outer.addLayout(left, 1)

        title = QLabel("3×3 Schiebe-Puzzel")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 14, QFont.Bold))
        left.addWidget(title)

        # Board
        self.board = QFrame()
        self.board.setObjectName("board")
        side = self.PAD * 2 + self.TILE * 3 + self.GAP * 2
        self.board.setFixedSize(side, side)
        self.board.setStyleSheet("""
            QFrame#board {
                background: #1f2937;
                border-radius: 16px;
            }
        """)
        left.addWidget(self.board, alignment=Qt.AlignCenter)

        # --- Controls: 3 Reihen ---
        controls = QVBoxLayout()
        left.addLayout(controls)

        # Reihe 1: Felder setzen + Eingabe + Setzen + Mischen
        r1 = QHBoxLayout()
        controls.addLayout(r1)

        r1.addStretch(1)
        r1.addWidget(QLabel("Felder setzen:"))
        self.input = QLineEdit("1 2 3 4 5 6 7 8 0")
        self.input.setPlaceholderText("z.B. 123456780")
        self.input.setMaximumWidth(120)
        r1.addWidget(self.input)

        self.btn_set = QPushButton("Setzen")
        self.btn_set.clicked.connect(self.on_set_state)
        r1.addWidget(self.btn_set)

        self.btn_shuffle = QPushButton("Mischen")
        self.btn_shuffle.clicked.connect(self.on_shuffle)
        r1.addWidget(self.btn_shuffle)
        r1.addStretch(1)

        # Reihe 2: Auto lösen + Stop + Reset + Log (Log ist eine Ebene runter)
        r2 = QHBoxLayout()
        controls.addLayout(r2)

        r2.addStretch(1)
        self.btn_solve = QPushButton("Auto lösen")
        self.btn_solve.clicked.connect(self.on_solve)
        r2.addWidget(self.btn_solve)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_stop.setEnabled(False)
        r2.addWidget(self.btn_stop)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)
        r2.addWidget(self.btn_reset)

        self.btn_log = QPushButton("Log anzeigen")
        self.btn_log.clicked.connect(self.toggle_log)
        r2.addWidget(self.btn_log)
        r2.addStretch(1)

        # Reihe 3: Prüfen + Bild laden + Bild löschen
        r3 = QHBoxLayout()
        controls.addLayout(r3)

        r3.addStretch(1)
        self.btn_check = QPushButton("Prüfen")
        self.btn_check.clicked.connect(self.on_check)
        r3.addWidget(self.btn_check)

        self.btn_img_load = QPushButton("Bild laden")
        self.btn_img_load.clicked.connect(self.on_load_image)
        r3.addWidget(self.btn_img_load)

        self.btn_img_clear = QPushButton("Bild löschen")
        self.btn_img_clear.clicked.connect(self.on_clear_image)
        self.btn_img_clear.setEnabled(False)
        r3.addWidget(self.btn_img_clear)
        r3.addStretch(1)

        # Buttons gleich groß
        self._set_buttons_equal_size([
            self.btn_set, self.btn_shuffle, self.btn_solve, self.btn_stop,
            self.btn_reset, self.btn_log, self.btn_check, self.btn_img_load, self.btn_img_clear
        ])

        self.status = QLabel("")
        self.status.setAlignment(Qt.AlignCenter)
        left.addWidget(self.status)
        left.addStretch(1)

        # Right/log panel
        self.log_panel = QFrame()
        self.log_panel.setObjectName("logpanel")
        self.log_panel.setStyleSheet("""
            QFrame#logpanel {
                background: #111827;
                border-radius: 12px;
                padding: 8px;
            }
            QLabel#logtitle {
                color: #e5e7eb;
                font-weight: 700;
            }
        """)
        self.log_panel.setFixedWidth(320)
        outer.addWidget(self.log_panel)

        lp = QVBoxLayout(self.log_panel)
        log_title = QLabel("Zug-Log")
        log_title.setObjectName("logtitle")
        lp.addWidget(log_title)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #0b1220;
                color: #e5e7eb;
                border: 1px solid #1f2937;
                border-radius: 10px;
                padding: 8px;
                font-family: Consolas, monospace;
                font-size: 12px;
            }
        """)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lp.addWidget(self.log_text, 1)

        self.btn_log_clear = QPushButton("Log leeren")
        self.btn_log_clear.clicked.connect(lambda: self.log_text.clear())
        self.btn_log_clear.setFixedSize(self.BTN_W, self.BTN_H)
        lp.addWidget(self.btn_log_clear)

    def _set_buttons_equal_size(self, buttons: List[QPushButton]):
        for b in buttons:
            b.setFixedSize(self.BTN_W, self.BTN_H)

    def _build_tiles(self):
        for val in range(1, 9):
            btn = QPushButton(str(val), self.board)
            btn.setObjectName("tile")
            btn.setFont(QFont("Arial", 18, QFont.Bold))
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton#tile {
                    background: #e5e7eb;
                    border: none;
                    border-radius: 14px;
                }
                QPushButton#tile:hover { background: #f3f4f6; }
                QPushButton#tile:pressed { background: #d1d5db; }
            """)
            btn.clicked.connect(lambda checked=False, v=val: self.on_tile_clicked(v))
            self.tiles[val] = btn

    # ---------- Helpers ----------

    def _refresh_base_size(self):
        """
        Setzt/merkt die Basisgröße (Log zu) zuverlässig, nachdem Qt das Layout berechnet hat.
        Du kannst BASE_SIZE ändern; diese Funktion sorgt nur dafür, dass resize() später klappt.
        """
        # Log zu erzwingen und dann Basisgröße als "richtig" merken
        was = self.log_panel.isVisible()
        self.log_panel.setVisible(False)
        self._base_size = QSize(self.BASE_SIZE)  # dein definierter Ausgangswert
        self.resize(self._base_size)
        self.log_panel.setVisible(was)

    def cell_rect(self, index: int) -> QRect:
        r, c = divmod(index, 3)
        x = self.PAD + c * (self.TILE + self.GAP)
        y = self.PAD + r * (self.TILE + self.GAP)
        return QRect(x, y, self.TILE, self.TILE)

    def idx_to_rc(self, idx: int) -> Tuple[int, int]:
        r, c = divmod(idx, 3)
        return (r + 1, c + 1)

    def _set_controls_enabled(self, enabled: bool):
        self.input.setEnabled(enabled)
        self.btn_set.setEnabled(enabled)
        self.btn_shuffle.setEnabled(enabled)
        self.btn_solve.setEnabled(enabled)
        self.btn_reset.setEnabled(enabled)
        self.btn_check.setEnabled(enabled)
        self.btn_img_load.setEnabled(enabled)
        self.btn_img_clear.setEnabled(enabled and self._image_mode)

        for b in self.tiles.values():
            b.setEnabled(enabled)

        # Log toggle darf immer
        self.btn_log.setEnabled(True)
        self.btn_log_clear.setEnabled(True)

    def _log(self, msg: str):
        self.log_text.append(msg)

    # ---------- Log / Fenstergröße ----------

    def toggle_log(self):
        vis = not self.log_panel.isVisible()
        if vis:
            self.log_panel.setVisible(True)
            self.btn_log.setText("Log verbergen")
            # Layout neu berechnen; Nutzer kann danach frei resizen
            self.adjustSize()
        else:
            self.log_panel.setVisible(False)
            self.btn_log.setText("Log anzeigen")
            # Wichtig: erst Layout anwenden, dann zurück auf Basisgröße
            QTimer.singleShot(0, lambda: self.resize(self._base_size))

    # ---------- Bild: Laden / Splitten / Anwenden ----------

    def _board_inner_side(self) -> int:
        return self.TILE * 3 + self.GAP * 2

    def on_load_image(self):
        if self._animating or self._auto_playing:
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Bild auswählen", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)"
        )
        if not path:
            return

        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.warning(self, "Fehler", "Konnte das Bild nicht laden.")
            return

        self._base_image = pm
        self._image_mode = True
        self.btn_img_clear.setEnabled(True)

        self._slice_image_into_tiles()
        self._apply_tile_appearance()
        self._log(f"--- BILD GELADEN: {path} ---")

    def on_clear_image(self):
        if self._animating or self._auto_playing:
            return
        self._image_mode = False
        self._base_image = None
        self._tile_images.clear()
        self.btn_img_clear.setEnabled(False)

        self._apply_tile_appearance()
        self._log("--- BILD GELÖSCHT: Standardoptik ---")

    def _slice_image_into_tiles(self):
        if not self._base_image or self._base_image.isNull():
            return

        pm = self._base_image
        side = min(pm.width(), pm.height())
        x0 = (pm.width() - side) // 2
        y0 = (pm.height() - side) // 2
        sq = pm.copy(x0, y0, side, side)

        inner = self._board_inner_side()
        scaled = sq.scaled(inner, inner, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self._tile_images.clear()
        for idx, val in enumerate(GOAL):
            if val == 0:
                continue
            r, c = divmod(idx, 3)
            x = c * (self.TILE + self.GAP)
            y = r * (self.TILE + self.GAP)
            self._tile_images[val] = scaled.copy(x, y, self.TILE, self.TILE)

    def _apply_tile_appearance(self):
        for val, btn in self.tiles.items():
            if self._image_mode and val in self._tile_images:
                btn.setText("")
                btn.setIcon(QIcon(self._tile_images[val]))
                btn.setIconSize(QSize(self.TILE, self.TILE))
                btn.setStyleSheet("""
                    QPushButton#tile {
                        background: transparent;
                        border: none;
                        border-radius: 14px;
                    }
                    QPushButton#tile:hover { background: rgba(255,255,255,0.08); }
                    QPushButton#tile:pressed { background: rgba(0,0,0,0.10); }
                """)
            else:
                btn.setIcon(QIcon())
                btn.setText(str(val))
                btn.setStyleSheet("""
                    QPushButton#tile {
                        background: #e5e7eb;
                        border: none;
                        border-radius: 14px;
                    }
                    QPushButton#tile:hover { background: #f3f4f6; }
                    QPushButton#tile:pressed { background: #d1d5db; }
                """)

    # ---------- Rendering / Animation ----------

    def _sync_tiles_to_state(self, animate: bool):
        self.status.setText("✅ Zielzustand erreicht!" if self.state == GOAL else "")

        if not animate:
            for idx, val in enumerate(self.state):
                if val == 0:
                    continue
                self.tiles[val].setGeometry(self.cell_rect(idx))
            return

        self._animating = True
        self._set_controls_enabled(False)

        group = QParallelAnimationGroup(self)
        self._anim_group = group
        moved_any = False

        for idx, val in enumerate(self.state):
            if val == 0:
                continue
            btn = self.tiles[val]
            target = self.cell_rect(idx)
            if btn.geometry() == target:
                continue

            anim = QPropertyAnimation(btn, b"geometry")
            anim.setDuration(self.ANIM_MS)
            anim.setEasingCurve(QEasingCurve.OutCubic)
            anim.setStartValue(btn.geometry())
            anim.setEndValue(target)
            group.addAnimation(anim)
            moved_any = True

        def done():
            self._animating = False
            if not self._auto_playing:
                self._set_controls_enabled(True)
            self.status.setText("✅ Zielzustand erreicht!" if self.state == GOAL else "")
            self._anim_group = None

            if self._auto_playing:
                QTimer.singleShot(self.PLAYBACK_GAP_MS, self._play_next_move)

        if moved_any:
            group.finished.connect(done)
            group.start()
        else:
            done()

    # ---------- Moves ----------

    def _apply_move_by_tile_value(self, tile_value: int, from_auto: bool):
        if self._animating:
            return

        zero_idx = self.state.index(0)
        tile_idx = self.state.index(tile_value)
        if tile_idx not in neighbors(zero_idx):
            return

        fr = self.idx_to_rc(tile_idx)
        to = self.idx_to_rc(zero_idx)
        self.state[zero_idx], self.state[tile_idx] = self.state[tile_idx], self.state[zero_idx]

        prefix = "AUTO" if from_auto else "USER"
        self._log(f"[{prefix}] {tile_value}  ({fr[0]},{fr[1]}) -> ({to[0]},{to[1]})")
        self._sync_tiles_to_state(animate=True)

    def on_tile_clicked(self, tile_value: int):
        if self._auto_playing:
            return
        self._apply_move_by_tile_value(tile_value, from_auto=False)

    # ---------- Buttons ----------

    def on_set_state(self):
        if self._animating or self._auto_playing:
            return

        vals = parse_state(self.input.text())
        if vals is None:
            QMessageBox.warning(self, "Ungültig", "Bitte genau 9 Zahlen 0–8 angeben (jede genau einmal).")
            return

        if not is_solvable_3x3(vals):
            res = QMessageBox.question(
                self, "Warnung: unlösbar",
                "Diese Ausgangslage ist (als 3×3) NICHT lösbar.\nTrotzdem setzen?",
                QMessageBox.Yes | QMessageBox.No
            )
            if res != QMessageBox.Yes:
                return

        self.state = vals
        self.initial_state = vals.copy()
        self._log(f"--- SET: {self.state} ---")
        self._sync_tiles_to_state(animate=True)

    def on_reset(self):
        if self._animating or self._auto_playing:
            return
        self.state = self.initial_state.copy()
        self._log(f"--- RESET: {self.state} ---")
        self._sync_tiles_to_state(animate=True)

    def on_check(self):
        QMessageBox.information(
            self, "Check",
            "✅ Puzzle ist gelöst." if self.state == GOAL else "❌ Noch nicht gelöst."
        )

    def on_shuffle(self):
        if self._animating or self._auto_playing:
            return

        self.state = GOAL.copy()
        zero_idx = self.state.index(0)
        last = None
        for _ in range(80):
            nbs = neighbors(zero_idx)
            if last is not None and last in nbs and len(nbs) > 1:
                nbs.remove(last)
            nxt = random.choice(nbs)
            self.state[zero_idx], self.state[nxt] = self.state[nxt], self.state[zero_idx]
            last = zero_idx
            zero_idx = nxt

        self.initial_state = self.state.copy()
        self.input.setText(" ".join(map(str, self.state)))
        self._log(f"--- SHUFFLE: {self.state} ---")
        self._sync_tiles_to_state(animate=True)

    def on_solve(self):
        if self._animating or self._auto_playing:
            return

        if not is_solvable_3x3(self.state):
            QMessageBox.warning(self, "Unlösbar", "Diese Ausgangslage ist unlösbar.")
            return

        self.status.setText("🧠 Suche nach Lösung …")
        QApplication.processEvents()

        moves = astar_solve(self.state)
        if moves is None:
            QMessageBox.warning(self, "Keine Lösung", "Keine Lösung gefunden (Limit erreicht).")
            self.status.setText("")
            return

        if len(moves) == 0:
            QMessageBox.information(self, "Fertig", "Ist schon gelöst 🙂")
            self.status.setText("✅ Zielzustand erreicht!")
            return

        self._log(f"--- AUTO SOLVE: {len(moves)} Züge ---")
        self._pending_moves = moves
        self._auto_playing = True

        self.btn_stop.setEnabled(True)
        self._set_controls_enabled(False)
        self.status.setText(f"▶️ Auto-Lösung läuft … (noch {len(self._pending_moves)} Züge)")
        self._play_next_move()

    def _play_next_move(self):
        if not self._auto_playing or self._animating:
            return

        if not self._pending_moves:
            self._auto_playing = False
            self.btn_stop.setEnabled(False)
            self._set_controls_enabled(True)
            self.status.setText("✅ Auto-Lösung fertig!" if self.state == GOAL else "⏹️ Auto-Lösung beendet.")
            return

        nxt = self._pending_moves.pop(0)
        self.status.setText(f"▶️ Auto-Lösung läuft … (noch {len(self._pending_moves)} Züge)")
        self._apply_move_by_tile_value(nxt, from_auto=True)

    def on_stop(self):
        if not self._auto_playing:
            return
        self._auto_playing = False
        self._pending_moves = []
        self.btn_stop.setEnabled(False)
        if not self._animating:
            self._set_controls_enabled(True)
        self.status.setText("⏹️ Auto-Lösung gestoppt.")


def main():
    app = QApplication(sys.argv)
    w = SlidingPuzzle()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
