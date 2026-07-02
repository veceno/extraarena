use crate::MAX_CANDIDATE_ACTIONS;

pub const NUM_HAND: usize = 4;
pub const NUM_BOARD: usize = 7;
pub const NUM_PLAY_POS: usize = 8;
pub const NUM_PLAY_TARGETS: usize = 17;
pub const NUM_ATTACK_TARGETS: usize = 8;
pub const PLAY_BASE: usize = 1;
pub const PLAY_STRIDE: usize = NUM_PLAY_POS * NUM_PLAY_TARGETS;
pub const ATTACK_BASE: usize = PLAY_BASE + NUM_HAND * PLAY_STRIDE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateAction {
    EndTurn,
    PlayCard {
        hand_index: usize,
        board_position: usize,
        target_code: usize,
    },
    Attack {
        attacker_index: usize,
        target_code: usize,
    },
}

pub fn decode_action_id(action_id: usize) -> Option<CandidateAction> {
    if action_id >= MAX_CANDIDATE_ACTIONS {
        return None;
    }
    if action_id == 0 {
        return Some(CandidateAction::EndTurn);
    }
    if (PLAY_BASE..ATTACK_BASE).contains(&action_id) {
        let flat = action_id - PLAY_BASE;
        let hand_flat = flat / PLAY_STRIDE;
        let inside = flat % PLAY_STRIDE;
        let board_position = inside / NUM_PLAY_TARGETS;
        let target_code = inside % NUM_PLAY_TARGETS;
        return Some(CandidateAction::PlayCard {
            hand_index: hand_flat,
            board_position,
            target_code,
        });
    }
    let flat = action_id - ATTACK_BASE;
    let attacker_index = flat / NUM_ATTACK_TARGETS;
    let target_code = flat % NUM_ATTACK_TARGETS;
    Some(CandidateAction::Attack {
        attacker_index,
        target_code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_action_layout_edges() {
        assert_eq!(decode_action_id(0), Some(CandidateAction::EndTurn));
        assert_eq!(
            decode_action_id(1),
            Some(CandidateAction::PlayCard {
                hand_index: 0,
                board_position: 0,
                target_code: 0,
            })
        );
        assert_eq!(
            decode_action_id(544),
            Some(CandidateAction::PlayCard {
                hand_index: 3,
                board_position: 7,
                target_code: 16,
            })
        );
        assert_eq!(
            decode_action_id(545),
            Some(CandidateAction::Attack {
                attacker_index: 0,
                target_code: 0,
            })
        );
        assert_eq!(
            decode_action_id(600),
            Some(CandidateAction::Attack {
                attacker_index: 6,
                target_code: 7,
            })
        );
        assert_eq!(decode_action_id(601), None);
    }
}
