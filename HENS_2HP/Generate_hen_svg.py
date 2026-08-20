def generate_hen_svg(results):
    """
    HEN grid diagram — SHI superstructure style.

    Layout rules
    ============
    * Hot streams run LEFT→RIGHT (red).  Tin on left, Tout on right.
    * Cold streams run RIGHT→LEFT (blue). Tin on right, Tout on left.
    * Stages are vertical column bands shared by all streams.
    * Each stream row is tall enough to fan out ALL its branches in every
      stage without vertical overlap.  Row height is computed from the
      maximum branch-count across all streams and stages.
    * A gap zone sits between the hot block and cold block; HEX connector
      lines and their annotation circles live entirely in that gap.
    * HEX symbol: a filled dot on the hot branch + a filled dot on the cold
      branch, connected by a straight vertical line.  A labeled circle
      (stream IDs + Q value) is placed at the midpoint of the connector line
      inside the gap zone.
    * Temperatures shown only on the flat mainline AFTER recombining and
      BEFORE splitting — i.e. at stage boundary nodes, never on branches.
    * Zero or near-zero Q heat exchangers and their splits are omitted.
    * Splits show exactly N branches for N active connections (no phantom line).
    * Stage numbers displayed at the very top of the diagram.
    * Fraction labels stacked vertically right after the split bar.
    * Final temperature line shown after utility symbols.

    Data conventions (from solve_hen_model)
    ========================================
    split_hot[i][k][j]  – fraction of hot stream i that goes to cold j in stage k
    split_cold[j][k][i] – fraction of cold stream j that comes from hot i in stage k
    T_hot[i][k]         – temperature of hot stream i at LEFT boundary of stage k
                          (k=0 → Tin, k=K → Tout)
    T_cold[j][k]        – temperature of cold stream j at LEFT boundary of stage k
                          (k=0 → Tout, k=K → Tin)
    hex_map[(hid,cid,s)]– heat exchanged (s = stage index 1-based)
    QH[j]               – hot utility on cold stream j
    QC[i]               – cold utility on hot stream i
    """

    HIDs       = results["HIDs"]
    CIDs       = results["CIDs"]
    T_hot      = results["T_hot"]
    T_cold     = results["T_cold"]
    Tout_H     = results["Tout_H"]
    Tout_C     = results["Tout_C"]
    split_hot  = results["split_hot"]
    split_cold = results["split_cold"]
    QH         = results["QH"]
    QC         = results["QC"]
    hex_map    = results["hex_map"]

    I = len(HIDs)
    J = len(CIDs)
    K = len(T_hot[0]) - 1   # number of stages

    Q_THRESH = 1.0   # heat duty threshold — ignore HEX below this (kW)

    # ── 1. Filter hex_map to remove near-zero Q entries ───────────────────
    hex_map_filtered = {key: Q for key, Q in hex_map.items() if Q > Q_THRESH}

    # ── 2. Compute maximum branch counts (only active HEX) ───────────────
    max_hot_branches = max(
        (sum(1 for j in range(J)
             if split_hot[i][k][j] > 1e-3
             and hex_map_filtered.get((HIDs[i], CIDs[j], k + 1), 0) > Q_THRESH)
         for i in range(I) for k in range(K)),
        default=1
    )
    max_cold_branches = max(
        (sum(1 for ii in range(I)
             if split_cold[j][k][ii] > 1e-3
             and hex_map_filtered.get((HIDs[ii], CIDs[j], k + 1), 0) > Q_THRESH)
         for j in range(J) for k in range(K)),
        default=1
    )

    # ── 3. Geometry constants ─────────────────────────────────────────────
    FONT        = "Arial, sans-serif"
    TEMP_SZ     = 10
    FRAC_SZ     = 10
    DOT_R       = 5
    HEX_CIRC_R  = 22      # radius of HEX annotation circle

    BRANCH_DY   = 36      # vertical pitch between branches within one stream
    STREAM_PAD  = 42      # extra vertical padding above/below branches within a row

    hot_row_h  = (max_hot_branches  - 1) * BRANCH_DY + 2 * STREAM_PAD + 20
    cold_row_h = (max_cold_branches - 1) * BRANCH_DY + 2 * STREAM_PAD + 20
    hot_row_h  = max(hot_row_h,  80)
    cold_row_h = max(cold_row_h, 80)

    GAP = max(2 * HEX_CIRC_R + 40, 80)

    PAD_L  = 160   # left margin (wider for Tin label + utility)
    PAD_R  = 160   # right margin
    PAD_T  = 80    # top margin (extra for stage labels at top)
    PAD_B  = 70    # bottom margin

    # Stage width: generous so split zones + HEX circles never overlap.
    # STUB is the flat mainline length on each side of the split zone.
    # We want:   STUB + split_zone + STUB  =  STAGE_W
    # split_zone must fit all HEX circles spread out horizontally.
    max_hex_per_stage = max(
        (sum(1 for i in range(I) for j in range(J)
             if hex_map_filtered.get((HIDs[i], CIDs[j], k + 1), 0) > Q_THRESH)
         for k in range(K)),
        default=1
    )
    STAGE_W = max(
        380,
        max_hex_per_stage * (2 * HEX_CIRC_R + 30) + 160,
        max_hot_branches * 50 + 160,
        max_cold_branches * 50 + 160,
    )

    # STUB: flat mainline before split and after merge.
    # Large enough that temp labels (shown at stage boundary) are clearly on
    # the flat segment, not inside the branch zone.
    STUB = max(60, int(STAGE_W * 0.22))

    W = PAD_L + K * STAGE_W + PAD_R
    H_hot  = I * hot_row_h
    H_cold = J * cold_row_h
    H = PAD_T + H_hot + GAP + H_cold + PAD_B

    y_hot = [PAD_T + i * hot_row_h + hot_row_h / 2  for i in range(I)]
    y_cold_base = PAD_T + H_hot + GAP
    y_cold = [y_cold_base + j * cold_row_h + cold_row_h / 2 for j in range(J)]

    x_node = [PAD_L + k * STAGE_W for k in range(K + 1)]

    y_gap_mid = PAD_T + H_hot + GAP / 2

    # ── 4. Pre-compute branch y-positions (active HEX only) ──────────────
    hot_branches  = [[[] for _ in range(K)] for _ in range(I)]
    cold_branches = [[[] for _ in range(K)] for _ in range(J)]

    for i in range(I):
        for k in range(K):
            active = [
                (j, split_hot[i][k][j])
                for j in range(J)
                if split_hot[i][k][j] > 1e-3
                and hex_map_filtered.get((HIDs[i], CIDs[j], k + 1), 0) > Q_THRESH
            ]
            n = len(active)
            if n == 0:
                hot_branches[i][k] = []
            elif n == 1:
                j, frac = active[0]
                hot_branches[i][k] = [(j, frac, y_hot[i])]
            else:
                # Fan out symmetrically — exactly n branches, no phantom centre line
                total_span = (n - 1) * BRANCH_DY
                y_start = y_hot[i] - total_span / 2
                hot_branches[i][k] = [
                    (active[idx][0], active[idx][1], y_start + idx * BRANCH_DY)
                    for idx in range(n)
                ]

    for j in range(J):
        for k in range(K):
            active = [
                (i2, split_cold[j][k][i2])
                for i2 in range(I)
                if split_cold[j][k][i2] > 1e-3
                and hex_map_filtered.get((HIDs[i2], CIDs[j], k + 1), 0) > Q_THRESH
            ]
            n = len(active)
            if n == 0:
                cold_branches[j][k] = []
            elif n == 1:
                i2, frac = active[0]
                cold_branches[j][k] = [(i2, frac, y_cold[j])]
            else:
                total_span = (n - 1) * BRANCH_DY
                y_start = y_cold[j] - total_span / 2
                cold_branches[j][k] = [
                    (active[idx][0], active[idx][1], y_start + idx * BRANCH_DY)
                    for idx in range(n)
                ]

    # ── 5. Build lookup: (i,j,k) → (y_hot_branch, y_cold_branch) ─────────
    hex_ys = {}
    for i in range(I):
        for k in range(K):
            for (j, frac, yb) in hot_branches[i][k]:
                hex_ys.setdefault((i, j, k), [None, None])[0] = yb
    for j in range(J):
        for k in range(K):
            for (i2, frac, yb) in cold_branches[j][k]:
                hex_ys.setdefault((i2, j, k), [None, None])[1] = yb

    # ── 6. Assign horizontal x positions for HEX annotation circles ───────
    # HEX circles MUST sit strictly inside the split zone:
    #   x_split_pt = x_node[k] + STUB   (where mainline fans out)
    #   x_merge_pt = x_node[k+1] - STUB (where branches rejoin)
    # So valid range is [x_split_pt + HEX_CIRC_R, x_merge_pt - HEX_CIRC_R].
    hex_x = {}
    for k in range(K):
        active_pairs = [(i, j) for (i, j, kk) in hex_ys if kk == k]
        active_pairs.sort()
        n = len(active_pairs)
        if n == 0:
            continue
        x_sp = x_node[k]     + STUB   # split point
        x_mp = x_node[k + 1] - STUB   # merge point
        x_lo = x_sp + HEX_CIRC_R + 5
        x_hi = x_mp - HEX_CIRC_R - 5
        if x_lo > x_hi:
            x_lo = x_hi = (x_sp + x_mp) / 2
        if n == 1:
            xs = [(x_lo + x_hi) / 2]
        else:
            step = (x_hi - x_lo) / (n - 1)
            xs = [x_lo + idx * step for idx in range(n)]
        for idx, (i, j) in enumerate(active_pairs):
            hex_x[(i, j, k)] = xs[idx]

    # ── 7. Build SVG ──────────────────────────────────────────────────────
    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{int(W)}" height="{int(H)}" '
        f'viewBox="0 0 {int(W)} {int(H)}" '
        f'style="background:#fdfdfd;font-family:{FONT};">'
    )

    svg.append("""<defs>
  <marker id="aR" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0,8 3,0 6" fill="#c0392b"/>
  </marker>
  <marker id="aL" markerWidth="8" markerHeight="6" refX="0" refY="3" orient="auto">
    <polygon points="8 0,0 3,8 6" fill="#1a5fa8"/>
  </marker>
</defs>""")

    HOT_C  = "#c0392b"
    COLD_C = "#1a5fa8"
    HEX_C  = "#7b2d8b"   # purple — distinct from stream colours, not green

    # Background bands
    svg.append(f'<rect x="0" y="{PAD_T}" width="{W}" height="{H_hot}" fill="#fff5f4"/>')
    svg.append(f'<rect x="0" y="{y_cold_base}" width="{W}" height="{H_cold}" fill="#f0f6ff"/>')
    svg.append(
        f'<rect x="0" y="{PAD_T + H_hot}" width="{W}" height="{GAP}" fill="#f9f5fb"/>'
    )

    # Stage vertical grid lines
    for k in range(K + 1):
        x = x_node[k]
        svg.append(
            f'<line x1="{x}" y1="{PAD_T}" x2="{x}" y2="{H - PAD_B}" '
            f'stroke="#dddddd" stroke-width="1"/>'
        )

    # Stage labels at the very top
    for k in range(K):
        xm = x_node[k] + STAGE_W / 2
        svg.append(
            f'<text x="{xm}" y="{PAD_T - 18}" text-anchor="middle" '
            f'fill="#555" font-size="14" font-weight="bold" font-style="italic">Stage {k+1}</text>'
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def dot_node(x, y, color, r=DOT_R):
        svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" '
            f'fill="{color}" stroke="white" stroke-width="1"/>'
        )

    def stream_label(x, y, text, color, anchor="end"):
        svg.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'fill="{color}" font-size="14" font-weight="bold">{text}</text>'
        )

    def temp_label(x, y, t, color, anchor="middle", dy=0):
        """Temperature label with °C unit."""
        svg.append(
            f'<text x="{x:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}" '
            f'fill="{color}" font-size="{TEMP_SZ}">{t:.0f} °C</text>'
        )

    def frac_lbl_stack(x_sp, branches, color, is_hot=True):
        """
        Draw split-fraction labels stacked vertically, aligned right after the
        split bar (at x_sp + small offset).  Each label sits on its branch's y.
        For hot streams the text is to the right of x_sp; for cold to the left.
        """
        x_lbl = x_sp + 8 if is_hot else x_sp - 8
        anchor = "start" if is_hot else "end"
        for (_, frac, yb) in branches:
            svg.append(
                f'<text x="{x_lbl:.1f}" y="{yb - 6:.1f}" text-anchor="{anchor}" '
                f'fill="{color}" font-size="{FRAC_SZ}" font-style="italic">{frac:.2f}</text>'
            )

    def hex_circle_lbl(cx, cy, hid, cid, Q):
        """Annotation circle in gap zone with MW units."""
        svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{HEX_CIRC_R}" '
            f'fill="white" stroke="{HEX_C}" stroke-width="2"/>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{cy - 5:.1f}" text-anchor="middle" '
            f'fill="{HEX_C}" font-size="8" font-weight="bold">H{hid}↔C{cid}</text>'
        )
        svg.append(
            f'<text x="{cx:.1f}" y="{cy + 6:.1f}" text-anchor="middle" '
            f'fill="#333" font-size="8">{Q:.1f} kW</text>'
        )

    def util_sym(cx, cy, color):
        """Crossed circle utility symbol."""
        r = 13
        svg.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" '
            f'fill="white" stroke="{color}" stroke-width="2"/>'
        )
        d = 9
        svg.append(
            f'<line x1="{cx-d:.1f}" y1="{cy-d:.1f}" x2="{cx+d:.1f}" y2="{cy+d:.1f}" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )
        svg.append(
            f'<line x1="{cx+d:.1f}" y1="{cy-d:.1f}" x2="{cx-d:.1f}" y2="{cy+d:.1f}" '
            f'stroke="{color}" stroke-width="1.8"/>'
        )

    # STUB is the flat mainline length before the split point and after the merge point.
    # Temperatures are drawn at stage-boundary nodes (x_node[k]) which sit on the
    # flat mainline — they are never inside the branch fan-out zone.
    # STUB must be long enough so the split bar is clearly separate from the node tick.

    # ════════════════════════════════════════════════════════════════════
    # HOT STREAMS
    # ════════════════════════════════════════════════════════════════════
    for i, hid in enumerate(HIDs):
        y0 = y_hot[i]
        x_start = PAD_L
        x_end   = x_node[K]

        stream_label(x_start - 10, y0, f"H{hid}", HOT_C, anchor="end")

        # Full backbone drawn first; branch segments drawn on top.
        svg.append(
            f'<line x1="{x_start}" y1="{y0:.1f}" x2="{x_end}" y2="{y0:.1f}" '
            f'stroke="{HOT_C}" stroke-width="2.5" marker-end="url(#aR)"/>'
        )

        # Tin label at stream start (flat, before any split)
        temp_label(x_start, y0, T_hot[i][0], HOT_C, anchor="middle", dy=-16)

        # Cold utility symbol at right end
        xu = x_end + 65
        # Line from network end to utility, with temperature label on this segment (before utility)
        svg.append(
            f'<line x1="{x_end}" y1="{y0:.1f}" x2="{xu - 13:.1f}" y2="{y0:.1f}" '
            f'stroke="{HOT_C}" stroke-width="2"/>'
        )
        # Temperature label on the line BEFORE the utility symbol
        temp_label((x_end + xu - 13) / 2, y0, T_hot[i][K], HOT_C, anchor="middle", dy=-10)
        #util_sym(xu, y0, HOT_C)
        qc_val = QC[i] if i < len(QC) else 0
        if qc_val > Q_THRESH:
            svg.append(
                f'<text x="{xu:.1f}" y="{y0 + 30:.1f}" text-anchor="middle" '
                f'fill="{HOT_C}" font-size="10">QC={qc_val:.1f} kW</text>'
            )
            util_sym(xu, y0, HOT_C)
        # Short line after utility symbol with the real Tout value
            x_final = xu + 13 + 40
            svg.append(
                f'<line x1="{xu + 13:.1f}" y1="{y0:.1f}" x2="{x_final:.1f}" y2="{y0:.1f}" '
                f'stroke="{HOT_C}" stroke-width="2"/>'
            )
            temp_label(x_final, y0, Tout_H[i], HOT_C, anchor="start", dy=-10)

        # Per-stage processing
        for k in range(K):
            branches = hot_branches[i][k]
            n = len(branches)
            if n == 0:
                # No HEX in this stage — just show stage-boundary temperature on flat line
                temp_label(x_node[k], y0, T_hot[i][k], HOT_C, anchor="middle", dy=-16)
                continue

            # Stage boundaries (the flat section nodes)
            x_stage_l = x_node[k]
            x_stage_r = x_node[k + 1]

            # Split/merge x-positions inset by STUB from stage boundaries
            x_sp = x_stage_l + STUB     # split bar position
            x_mp = x_stage_r - STUB     # merge bar position

            # ── Temperature labels on the FLAT mainline (at stage boundary nodes) ──
            # Left boundary: after recombine from previous stage (or Tin for k=0)
            temp_label(x_stage_l, y0, T_hot[i][k],     HOT_C, anchor="middle", dy=-16)
            # Right boundary: before split into next stage
            if k == K - 1:
                pass  # last stage right boundary = x_end; Tout shown before utility above
            else:
                temp_label(x_stage_r, y0, T_hot[i][k + 1], HOT_C, anchor="middle", dy=-16)

            if n == 1:
                # Single branch — HEX dot on mainline; no visual split needed
                j_idx, frac, yb = branches[0]
                xh = hex_x.get((i, j_idx, k), (x_sp + x_mp) / 2)
                dot_node(xh, yb, HOT_C)
            else:
                # ── True N-way split ──
                y_vals = [yb for (_, _, yb) in branches]
                y_top  = min(y_vals)
                y_bot  = max(y_vals)

                # Erase backbone through the split zone so branches are clean
                svg.append(
                    f'<rect x="{x_sp:.1f}" y="{y0 - 3:.1f}" '
                    f'width="{x_mp - x_sp:.1f}" height="6" fill="#fff5f4"/>'
                )

                # Flat stubs: stage_l → split_bar  and  merge_bar → stage_r
                svg.append(
                    f'<line x1="{x_stage_l:.1f}" y1="{y0:.1f}" x2="{x_sp:.1f}" y2="{y0:.1f}" '
                    f'stroke="{HOT_C}" stroke-width="2.5"/>'
                )
                svg.append(
                    f'<line x1="{x_mp:.1f}" y1="{y0:.1f}" x2="{x_stage_r:.1f}" y2="{y0:.1f}" '
                    f'stroke="{HOT_C}" stroke-width="2.5"/>'
                )

                # Vertical split bar
                svg.append(
                    f'<line x1="{x_sp:.1f}" y1="{y_top:.1f}" '
                    f'x2="{x_sp:.1f}" y2="{y_bot:.1f}" '
                    f'stroke="{HOT_C}" stroke-width="2"/>'
                )
                # Vertical merge bar
                svg.append(
                    f'<line x1="{x_mp:.1f}" y1="{y_top:.1f}" '
                    f'x2="{x_mp:.1f}" y2="{y_bot:.1f}" '
                    f'stroke="{HOT_C}" stroke-width="2"/>'
                )

                # Branch lines and HEX dots
                for (j_idx, frac, yb) in branches:
                    xh = hex_x.get((i, j_idx, k), (x_sp + x_mp) / 2)

                    dot_node(x_sp, yb, HOT_C, r=3)
                    dot_node(x_mp, yb, HOT_C, r=3)

                    svg.append(
                        f'<line x1="{x_sp:.1f}" y1="{yb:.1f}" '
                        f'x2="{xh:.1f}" y2="{yb:.1f}" '
                        f'stroke="{HOT_C}" stroke-width="1.8"/>'
                    )
                    svg.append(
                        f'<line x1="{xh:.1f}" y1="{yb:.1f}" '
                        f'x2="{x_mp:.1f}" y2="{yb:.1f}" '
                        f'stroke="{HOT_C}" stroke-width="1.8"/>'
                    )

                    dot_node(xh, yb, HOT_C)

                # Fraction labels stacked right after the split bar
                frac_lbl_stack(x_sp, branches, HOT_C, is_hot=True)

    # ════════════════════════════════════════════════════════════════════
    # COLD STREAMS
    # ════════════════════════════════════════════════════════════════════
    for j, cid in enumerate(CIDs):
        y0 = y_cold[j]
        x_start = PAD_L
        x_end   = x_node[K]

        stream_label(x_end + 10, y0, f"C{cid}", COLD_C, anchor="start")

        # Full backbone (arrow on left end = start of cold flow = low-temp side)
        svg.append(
            f'<line x1="{x_start}" y1="{y0:.1f}" x2="{x_end}" y2="{y0:.1f}" '
            f'stroke="{COLD_C}" stroke-width="2.5" marker-start="url(#aL)"/>'
        )

        # Tin label at right end (flat, before any split from cold's perspective)
        temp_label(x_end, y0, T_cold[j][K], COLD_C, anchor="middle", dy=20)

        # Hot utility at left end
        xu = x_start - 65
        # Line from network start to utility, with temperature label on this segment (before utility)
        svg.append(
            f'<line x1="{xu + 13:.1f}" y1="{y0:.1f}" x2="{x_start}" y2="{y0:.1f}" '
            f'stroke="{COLD_C}" stroke-width="2"/>'
        )
        # Temperature label on the line BEFORE the utility symbol (cold flows right→left, so
        # the utility is on the left = outlet side; the temp here is T_cold[j][0])
        temp_label((xu + 13 + x_start) / 2, y0, T_cold[j][0], COLD_C, anchor="middle", dy=16)
        #util_sym(xu, y0, COLD_C)
        qh_val = QH[j] if j < len(QH) else 0
        if qh_val > Q_THRESH:
            svg.append(
                f'<text x="{xu:.1f}" y="{y0 + 30:.1f}" text-anchor="middle" '
                f'fill="{COLD_C}" font-size="10">QH={qh_val:.1f} kW</text>'
            )
            util_sym(xu, y0, COLD_C)
        # Short line after utility symbol with the real Tout value
            x_final = xu - 13 - 40
            svg.append(
                f'<line x1="{x_final:.1f}" y1="{y0:.1f}" x2="{xu - 13:.1f}" y2="{y0:.1f}" '
                f'stroke="{COLD_C}" stroke-width="2"/>'
            )
            temp_label(x_final, y0, Tout_C[j], COLD_C, anchor="end", dy=16)

        # Per-stage processing
        # Cold flows RIGHT→LEFT so "entry" is the right boundary, "exit" is left.
        for k in range(K):
            branches = cold_branches[j][k]
            n = len(branches)

            x_stage_r = x_node[k + 1]   # right boundary (cold entry side)
            x_stage_l = x_node[k]       # left boundary  (cold exit side)

            # Split/merge inset by STUB
            x_sp = x_stage_r - STUB     # split bar (cold splits here, coming from right)
            x_mp = x_stage_l + STUB     # merge bar (cold recombines here, exiting left)

            if n == 0:
                # No HEX this stage — show stage-boundary temperature on flat line
                temp_label(x_stage_r, y0, T_cold[j][k + 1], COLD_C, anchor="middle", dy=20)
                continue

            # ── Temperature labels on FLAT mainline at stage boundary nodes ──
            # Right boundary: flat mainline before cold splits (Tin side for this stage)
            temp_label(x_stage_r, y0, T_cold[j][k + 1], COLD_C, anchor="middle", dy=20)
            # Left boundary: flat mainline after cold recombines (Tout side, shown after utility for k=0)
            if k == 0:
                pass  # leftmost boundary = x_start; temp shown before utility above
            else:
                temp_label(x_stage_l, y0, T_cold[j][k], COLD_C, anchor="middle", dy=20)

            if n == 1:
                i_idx, frac, yb = branches[0]
                xh = hex_x.get((i_idx, j, k), (x_mp + x_sp) / 2)
                dot_node(xh, yb, COLD_C)
            else:
                # ── True N-way split ──
                y_vals = [yb for (_, _, yb) in branches]
                y_top  = min(y_vals)
                y_bot  = max(y_vals)

                # Erase backbone through split zone
                svg.append(
                    f'<rect x="{x_mp:.1f}" y="{y0 - 3:.1f}" '
                    f'width="{x_sp - x_mp:.1f}" height="6" fill="#f0f6ff"/>'
                )

                # Flat stubs
                svg.append(
                    f'<line x1="{x_sp:.1f}" y1="{y0:.1f}" x2="{x_stage_r:.1f}" y2="{y0:.1f}" '
                    f'stroke="{COLD_C}" stroke-width="2.5"/>'
                )
                svg.append(
                    f'<line x1="{x_stage_l:.1f}" y1="{y0:.1f}" x2="{x_mp:.1f}" y2="{y0:.1f}" '
                    f'stroke="{COLD_C}" stroke-width="2.5"/>'
                )

                # Vertical split bar (right side — cold entry)
                svg.append(
                    f'<line x1="{x_sp:.1f}" y1="{y_top:.1f}" '
                    f'x2="{x_sp:.1f}" y2="{y_bot:.1f}" '
                    f'stroke="{COLD_C}" stroke-width="2"/>'
                )
                # Vertical merge bar (left side — cold exit)
                svg.append(
                    f'<line x1="{x_mp:.1f}" y1="{y_top:.1f}" '
                    f'x2="{x_mp:.1f}" y2="{y_bot:.1f}" '
                    f'stroke="{COLD_C}" stroke-width="2"/>'
                )

                for (i_idx, frac, yb) in branches:
                    xh = hex_x.get((i_idx, j, k), (x_mp + x_sp) / 2)

                    dot_node(x_sp, yb, COLD_C, r=3)
                    dot_node(x_mp, yb, COLD_C, r=3)

                    svg.append(
                        f'<line x1="{x_mp:.1f}" y1="{yb:.1f}" '
                        f'x2="{xh:.1f}" y2="{yb:.1f}" '
                        f'stroke="{COLD_C}" stroke-width="1.8"/>'
                    )
                    svg.append(
                        f'<line x1="{xh:.1f}" y1="{yb:.1f}" '
                        f'x2="{x_sp:.1f}" y2="{yb:.1f}" '
                        f'stroke="{COLD_C}" stroke-width="1.8"/>'
                    )

                    dot_node(xh, yb, COLD_C)

                # Fraction labels stacked right after the split bar (cold: left of x_sp)
                frac_lbl_stack(x_sp, branches, COLD_C, is_hot=False)

    # ════════════════════════════════════════════════════════════════════
    # HEX CONNECTOR LINES  (vertical, hot dot → cold dot, through gap)
    # ════════════════════════════════════════════════════════════════════
    for (i, j, k), ys in hex_ys.items():
        hid = HIDs[i]
        cid = CIDs[j]
        qval = hex_map_filtered.get((hid, cid, k + 1), 0)
        if qval <= Q_THRESH:
            continue

        y_h, y_c = ys
        y_h = y_h if y_h is not None else y_hot[i]
        y_c = y_c if y_c is not None else y_cold[j]

        xh = hex_x.get((i, j, k), x_node[k] + STAGE_W / 2)

        svg.append(
            f'<line x1="{xh:.1f}" y1="{y_h:.1f}" x2="{xh:.1f}" y2="{y_c:.1f}" '
            f'stroke="{HEX_C}" stroke-width="1.8" stroke-dasharray="5,3"/>'
        )

        y_circ = y_gap_mid
        hex_circle_lbl(xh, y_circ, hid, cid, qval)

    # ════════════════════════════════════════════════════════════════════
    # SECTION DIVIDERS in gap
    # ════════════════════════════════════════════════════════════════════
    y_div = PAD_T + H_hot + 2
    svg.append(
        f'<line x1="0" y1="{y_div:.1f}" x2="{W}" y2="{y_div:.1f}" '
        f'stroke="#cccccc" stroke-width="1" stroke-dasharray="8,4"/>'
    )
    y_div2 = PAD_T + H_hot + GAP - 2
    svg.append(
        f'<line x1="0" y1="{y_div2:.1f}" x2="{W}" y2="{y_div2:.1f}" '
        f'stroke="#cccccc" stroke-width="1" stroke-dasharray="8,4"/>'
    )

    # ════════════════════════════════════════════════════════════════════
    # LEGEND
    # ════════════════════════════════════════════════════════════════════
    ly = H - 30
    items = [
        (HOT_C,  "━━━▶", "Hot stream"),
        (COLD_C, "◀━━━", "Cold stream"),
        (HEX_C,  "●",     "HEX connector (kW)"),
        (COLD_C, "⊗",     "Utility (QH / QC, kW)"),
    ]
    x_leg = 20
    for col, sym, lbl in items:
        svg.append(f'<text x="{x_leg}" y="{ly}" fill="{col}" font-size="13">{sym}</text>')
        svg.append(f'<text x="{x_leg + 50}" y="{ly}" fill="#666" font-size="11">{lbl}</text>')
        x_leg += 210

    svg.append("</svg>")
    return "\n".join(svg)