// Ability Effect Categories for RL Observation Encoding
//
// Categories are split into:
// - EFFECT types: What the ability does (Heals, Charge, Damage, etc.)
// - ACTIVATION types: When/how often it can be used (Activated, Passive, etc.)

use crate::ability_ids::AbilityId;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbilityEffectCategory {
    // === EFFECT TYPES ===
    Heals,
    Charge,
    Damage,
    Switch,
    Protect,
    Buff,
    EnergyManip,
    Evolution,
    Draw,
    Disrupt,
    Debuff,
    Drawback, // Pokémon take damage, end turn, etc

    // === ACTIVATION TYPES ===
    Passive,
    Activated,
    OncePerGame,
    Triggered,
    OncePerTurn,
}

pub const NUM_ABILITY_EFFECT_CATEGORIES: usize = 17;

pub fn get_ability_effect_categories(ability_id: AbilityId) -> &'static [AbilityEffectCategory] {
    use AbilityEffectCategory::*;

    match ability_id {
        // === HEALS ===
        AbilityId::A2022ShayminFragrantFlowerGarden => &[Heals, Activated, OncePerTurn],
        AbilityId::A4a022MiloticHealingRipples => &[Heals, Triggered, OncePerGame],
        AbilityId::A4083EspeonExPsychicHealing => &[Heals, Activated, OncePerTurn],
        AbilityId::PA037CresseliaExLunarPlumage => &[Heals, Triggered],
        AbilityId::A2072DusknoirShadowVoid => &[Heals, Triggered, Drawback],
        AbilityId::B1121IndeedeeExWatchOver => &[Heals, Activated, OncePerTurn],
        AbilityId::A1007Butterfree => &[Heals, Activated, OncePerTurn],

        // === CHARGE ===
        AbilityId::A1098MagnetonVoltCharge => &[Charge, Activated, OncePerTurn],
        AbilityId::A1132Gardevoir => &[Charge, Activated, OncePerTurn],
        AbilityId::A2a010LeafeonExForestBreath => &[Charge, Activated, OncePerTurn],
        AbilityId::A3b009FlareonExCombust => &[Charge, Activated, OncePerTurn, Drawback],
        AbilityId::B1a012CharmeleonIgnition => &[Charge, OncePerGame, Triggered],
        AbilityId::A2b035GiratinaExBrokenSpaceBellow => &[Charge, Activated, OncePerTurn, Drawback],
        AbilityId::A3a021ZeraoraThunderclapFlash => &[Charge, Triggered],
        AbilityId::B1157HydreigonRoarInUnison => &[Charge, Activated, OncePerTurn, Drawback],

        // === DRAW ===
        AbilityId::A4a010EnteiExLegendaryPulse => &[Draw, Passive],
        AbilityId::A4a020SuicuneExLegendaryPulse => &[Draw, Passive],
        AbilityId::A4a025RaikouExLegendaryPulse => &[Draw, Passive],
        AbilityId::A3b034SylveonExHappyRibbon => &[Draw, Triggered, OncePerGame],
        AbilityId::A3a027ShiinoticIlluminate => &[Draw, Activated, OncePerTurn],

        // === DAMAGE ===
        AbilityId::A1089GreninjaWaterShuriken => &[Damage, Activated, OncePerTurn],
        AbilityId::A1177Weezing => &[Damage, Debuff, Activated, OncePerTurn],
        AbilityId::A2a050CrobatCunningLink => &[Damage, Activated, OncePerTurn],
        AbilityId::A1061PoliwrathCounterattack => &[Damage, Triggered],
        AbilityId::A2110DarkraiExNightmareAura => &[Damage, Triggered],

        // === SWITCH ===
        AbilityId::A1188PidgeotDriveOff => &[Switch, Activated, OncePerTurn],
        AbilityId::A1020VictreebelFragranceTrap => &[Switch, Activated, OncePerTurn],
        AbilityId::A3122SolgaleoExRisingRoad => &[Switch, Activated, OncePerTurn],
        AbilityId::A4112UmbreonExDarkChase => &[Switch, Activated, OncePerTurn],
        AbilityId::A3a062CelesteelaUltraThrusters => &[Switch, Activated],

        // === PROTECT ===
        AbilityId::A2a022GlaceonExSnowyTerrain => &[Protect, Passive],
        AbilityId::A3066OricoricSafeguard => &[Protect, Passive],
        AbilityId::A3141KomalaComatose => &[Protect, Passive],
        AbilityId::A3b057SnorlaxExFullMouthManner => &[Protect, Passive],
        AbilityId::B1a065FurfrouFurCoat => &[Protect, Passive],
        AbilityId::B1a018WartortleShellShield => &[Protect, Passive],
        AbilityId::A2a071Arceus => &[Protect, Passive],

        // === BUFF ===
        AbilityId::A2092LucarioFightingCoach => &[Buff, Passive],
        AbilityId::B1172AegislashCursedMetal => &[Buff, Passive],
        AbilityId::A2a069ShayminSkySupport => &[Buff, Activated],
        AbilityId::B1a034ReuniclusInfiniteIncrease => &[Buff, Triggered],
        AbilityId::A2078GiratinaLevitate => &[Buff, Passive],

        // === DEBUFF ===
        AbilityId::A3a015LuxrayIntimidatingFang => &[Debuff, Passive],
        AbilityId::A3a042NihilegoMorePoison => &[Debuff, Passive],
        AbilityId::B1160DragalgeExPoisonPoint => &[Debuff, Triggered],

        // === DISRUPT ===
        AbilityId::A1a046AerodactylExPrimevalLaw => &[Disrupt, Passive],
        AbilityId::B1177GoomyStickyMembrane => &[Disrupt, Passive],
        AbilityId::A1123GengarExShadowySpellbind => &[Disrupt, Passive],
        AbilityId::B1a006AriadosTrapTerritory => &[Disrupt, Passive],

        // === ENERGY MANIPULATION ===
        AbilityId::A1a019VaporeonWashOut => &[EnergyManip, Triggered],
        AbilityId::A1a006SerperiorJungleTotem => &[EnergyManip, Passive],
        AbilityId::B1073GreninjaExShiftingStream => &[EnergyManip, Activated, OncePerTurn],

        // === EVOLUTION ===
        AbilityId::A3b056EeveeExVeeveeVolve => &[Evolution, Passive],
        AbilityId::B1184EeveeBoostedEvolution => &[Evolution, Passive],

        // === INFO (no useful effect for agent) ===
        AbilityId::A4a032MisdreavusInfiltratingInspection => &[Activated, OncePerTurn],
    }
}

pub fn encode_ability_categories(ability_id: AbilityId) -> [f32; NUM_ABILITY_EFFECT_CATEGORIES] {
    let categories = get_ability_effect_categories(ability_id);
    let mut encoding = [0.0; NUM_ABILITY_EFFECT_CATEGORIES];
    for cat in categories {
        encoding[*cat as usize] = 1.0;
    }
    encoding
}

pub fn encode_ability_from_card_id(card_id: &str) -> [f32; NUM_ABILITY_EFFECT_CATEGORIES] {
    if let Some(ability_id) = AbilityId::from_pokemon_id(card_id) {
        encode_ability_categories(ability_id)
    } else {
        [0.0; NUM_ABILITY_EFFECT_CATEGORIES]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magneton_is_charge_activated() {
        let cats = get_ability_effect_categories(AbilityId::A1098MagnetonVoltCharge);
        assert!(cats.contains(&AbilityEffectCategory::Charge));
        assert!(cats.contains(&AbilityEffectCategory::Activated));
    }

    #[test]
    fn test_oricoric_is_protect_passive() {
        let cats = get_ability_effect_categories(AbilityId::A3066OricoricSafeguard);
        assert!(cats.contains(&AbilityEffectCategory::Protect));
        assert!(cats.contains(&AbilityEffectCategory::Passive));
    }

    #[test]
    fn test_encoding_size() {
        let encoding = encode_ability_categories(AbilityId::A1098MagnetonVoltCharge);
        assert_eq!(encoding.len(), NUM_ABILITY_EFFECT_CATEGORIES);
    }

    #[test]
    fn test_all_abilities_covered() {
        use AbilityId::*;
        let all_abilities = [
            A1020VictreebelFragranceTrap,
            A1061PoliwrathCounterattack,
            A1089GreninjaWaterShuriken,
            A1098MagnetonVoltCharge,
            A1123GengarExShadowySpellbind,
            A1177Weezing,
            A1188PidgeotDriveOff,
            A1007Butterfree,
            A1132Gardevoir,
            A1a006SerperiorJungleTotem,
            A1a046AerodactylExPrimevalLaw,
            A1a019VaporeonWashOut,
            A2a010LeafeonExForestBreath,
            A2a050CrobatCunningLink,
            A2a071Arceus,
            A2022ShayminFragrantFlowerGarden,
            A2072DusknoirShadowVoid,
            A2078GiratinaLevitate,
            A2092LucarioFightingCoach,
            A2a069ShayminSkySupport,
            A2110DarkraiExNightmareAura,
            A2b035GiratinaExBrokenSpaceBellow,
            A3066OricoricSafeguard,
            A3122SolgaleoExRisingRoad,
            A3141KomalaComatose,
            A3a015LuxrayIntimidatingFang,
            A3a021ZeraoraThunderclapFlash,
            A3a027ShiinoticIlluminate,
            A3a042NihilegoMorePoison,
            A3a062CelesteelaUltraThrusters,
            A3b009FlareonExCombust,
            A3b034SylveonExHappyRibbon,
            A3b056EeveeExVeeveeVolve,
            A3b057SnorlaxExFullMouthManner,
            A4083EspeonExPsychicHealing,
            A4112UmbreonExDarkChase,
            A4a010EnteiExLegendaryPulse,
            A4a020SuicuneExLegendaryPulse,
            A4a022MiloticHealingRipples,
            A4a025RaikouExLegendaryPulse,
            B1073GreninjaExShiftingStream,
            B1121IndeedeeExWatchOver,
            B1157HydreigonRoarInUnison,
            B1160DragalgeExPoisonPoint,
            B1184EeveeBoostedEvolution,
            B1172AegislashCursedMetal,
            B1177GoomyStickyMembrane,
            PA037CresseliaExLunarPlumage,
            B1a006AriadosTrapTerritory,
            B1a012CharmeleonIgnition,
            B1a018WartortleShellShield,
            B1a034ReuniclusInfiniteIncrease,
            B1a065FurfrouFurCoat,
        ];
        for ability in all_abilities {
            let _ = get_ability_effect_categories(ability);
        }
    }
}
