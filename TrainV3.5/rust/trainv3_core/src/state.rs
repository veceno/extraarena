#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CardType {
    Hero,
    Warrior,
    Potion,
}

#[derive(Debug, Clone)]
pub struct CardShapeInput<'a> {
    pub card_type: CardType,
    pub mana_cost: i32,
    pub attack: i32,
    pub hp: i32,
    pub max_hp: i32,
    pub is_ready: bool,
    pub is_frozen: bool,
    pub level: i32,
    pub mechanics: &'a [&'a str],
}

impl<'a> CardShapeInput<'a> {
    pub fn hp_fraction(&self) -> f32 {
        if self.max_hp <= 0 {
            0.0
        } else {
            (self.hp as f64 / self.max_hp as f64) as f32
        }
    }
}
