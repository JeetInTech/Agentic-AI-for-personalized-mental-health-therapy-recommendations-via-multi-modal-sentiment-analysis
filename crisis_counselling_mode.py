"""
Crisis Counselling Mode - Advanced Emotional Support System
Provides compassionate, context-aware, and adaptive responses for users in distress

Handles: COVID-19 stress, loss/grief, breakups, depression, anxiety, trauma, and more
"""

import logging
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrisisType(Enum):
    """Types of crisis situations"""
    GRIEF_LOSS = "grief_loss"
    RELATIONSHIP_BREAKUP = "relationship_breakup"
    COVID_STRESS = "covid_stress"
    DEPRESSION = "depression"
    ANXIETY_PANIC = "anxiety_panic"
    TRAUMA = "trauma"
    SUICIDAL_IDEATION = "suicidal_ideation"
    SELF_HARM = "self_harm"
    ISOLATION_LONELINESS = "isolation_loneliness"
    HEALTH_CRISIS = "health_crisis"
    FINANCIAL_STRESS = "financial_stress"
    FAMILY_CONFLICT = "family_conflict"
    GENERAL_DISTRESS = "general_distress"


class EmotionalTone(Enum):
    """Emotional tone levels for adaptive responses"""
    IMMEDIATE_CRISIS = "immediate_crisis"  # Suicidal, self-harm
    ACUTE_DISTRESS = "acute_distress"      # High emotional pain
    MODERATE_CONCERN = "moderate_concern"   # Struggling but stable
    SUPPORTIVE_GROWTH = "supportive_growth" # Recovery/progress


class CrisisCounsellingMode:
    """
    Advanced crisis counselling system with emotional intelligence
    and context-aware adaptive responses
    """

    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)

        # Track conversation context
        self.conversation_history = []
        self.identified_crisis_type = None
        self.emotional_tone = EmotionalTone.MODERATE_CONCERN
        self.user_context = {
            'mentioned_topics': set(),
            'expressed_emotions': [],
            'coping_attempts': [],
            'support_system': None,
            'risk_level': 'low'
        }

        # Initialize crisis detection patterns
        self._init_crisis_patterns()

        # Initialize response templates
        self._init_response_templates()

        # Initialize coping strategies
        self._init_coping_strategies()

        # Professional resources
        self._init_professional_resources()

        logger.info("Crisis Counselling Mode initialized")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def _init_crisis_patterns(self):
        """Initialize patterns for detecting different crisis types"""
        self.crisis_patterns = {
            CrisisType.SUICIDAL_IDEATION: {
                'keywords': [
                    'kill myself', 'end my life', 'want to die', 'suicide',
                    'better off dead', 'end it all', 'no reason to live',
                    'take my own life', 'not worth living'
                ],
                'severity': 'critical',
                'immediate_response': True
            },
            CrisisType.SELF_HARM: {
                'keywords': [
                    'cut myself', 'hurt myself', 'self harm', 'self-harm',
                    'cutting', 'burning myself', 'harm myself', 'injure myself'
                ],
                'severity': 'critical',
                'immediate_response': True
            },
            CrisisType.GRIEF_LOSS: {
                'keywords': [
                    'died', 'passed away', 'death', 'lost', 'funeral',
                    'grief', 'mourning', 'miss them', 'gone forever',
                    'never see again', 'left us', 'departed'
                ],
                'severity': 'high',
                'immediate_response': False
            },
            CrisisType.RELATIONSHIP_BREAKUP: {
                'keywords': [
                    'broke up', 'breakup', 'break up', 'ex boyfriend',
                    'ex girlfriend', 'ended relationship', 'dumped me',
                    'left me', 'relationship over', 'we split'
                ],
                'severity': 'moderate',
                'immediate_response': False
            },
            CrisisType.COVID_STRESS: {
                'keywords': [
                    'covid', 'coronavirus', 'pandemic', 'quarantine',
                    'lockdown', 'isolated', 'can\'t go out', 'social distancing',
                    'lost job covid', 'long covid'
                ],
                'severity': 'moderate',
                'immediate_response': False
            },
            CrisisType.DEPRESSION: {
                'keywords': [
                    'depressed', 'depression', 'hopeless', 'empty',
                    'numb', 'no energy', 'can\'t get out of bed',
                    'worthless', 'meaningless', 'no motivation'
                ],
                'severity': 'high',
                'immediate_response': False
            },
            CrisisType.ANXIETY_PANIC: {
                'keywords': [
                    'panic attack', 'anxiety', 'can\'t breathe', 'racing heart',
                    'terrified', 'panic', 'anxious', 'constant worry',
                    'overwhelming fear', 'chest tight'
                ],
                'severity': 'high',
                'immediate_response': False
            },
            CrisisType.TRAUMA: {
                'keywords': [
                    'trauma', 'ptsd', 'flashback', 'triggered', 'abuse',
                    'assault', 'attacked', 'traumatic', 'nightmares',
                    'violated', 'hurt by'
                ],
                'severity': 'high',
                'immediate_response': False
            },
            CrisisType.ISOLATION_LONELINESS: {
                'keywords': [
                    'lonely', 'alone', 'no friends', 'isolated', 'nobody cares',
                    'no one understands', 'all by myself', 'completely alone',
                    'no one to talk to'
                ],
                'severity': 'moderate',
                'immediate_response': False
            },
            CrisisType.HEALTH_CRISIS: {
                'keywords': [
                    'diagnosed', 'illness', 'sick', 'disease', 'medical',
                    'hospital', 'health scare', 'terminal', 'cancer',
                    'chronic pain'
                ],
                'severity': 'high',
                'immediate_response': False
            },
            CrisisType.FINANCIAL_STRESS: {
                'keywords': [
                    'money problems', 'can\'t afford', 'debt', 'bills',
                    'bankrupt', 'lost job', 'unemployed', 'evicted',
                    'foreclosure', 'financial crisis'
                ],
                'severity': 'moderate',
                'immediate_response': False
            }
        }

    def _init_response_templates(self):
        """Initialize empathetic response templates for each crisis type"""
        self.response_templates = {
            CrisisType.SUICIDAL_IDEATION: {
                EmotionalTone.IMMEDIATE_CRISIS: [
                    "I'm so glad you reached out. What you're feeling right now matters deeply, and I want you to know that you're not alone. Your life has value, even when it doesn't feel that way. Can we talk about what's happening right now? And please know that immediate help is available - AASRA is there 24/7 at 9820466726, or you can reach Vandrevala Foundation at 1860-2662-345 (9am-9pm). You deserve support through this.",
                    "Thank you for trusting me with these incredibly difficult feelings. Right now, your safety is what matters most. I want you to know that these feelings, as overwhelming as they are, can change - and there are people who want to help you through this. Please reach out to AASRA (9820466726) or 112 if you're in immediate danger. Can you tell me if you're somewhere safe right now?",
                    "I hear how much pain you're in, and I'm really concerned about you. What you're going through is incredibly hard, but you took an important step by reaching out. Please don't face this alone - call AASRA at 9820466726 right now for immediate support, or 112 if you need emergency help. While we talk, can you tell me if there's someone nearby who can be with you?"
                ]
            },
            CrisisType.SELF_HARM: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "I can hear how much emotional pain you're carrying, and I'm really concerned about you. Self-harm often happens when the pain inside feels unbearable and there seems to be no other way to cope. You deserve care and support, not harm. Can we talk about what's happening? iCall Psychosocial Helpline (9152987821) can provide immediate support Monday-Saturday 8am-10pm. Are you somewhere safe right now?",
                    "Thank you for sharing this with me - that takes courage. When we feel overwhelmed, sometimes our mind searches for any way to release the pain. But you deserve healing, not hurt. There are other ways to manage these intense feelings, and people who can help you find them. Can we explore what you're feeling right now? And please know AASRA (9820466726) is available 24/7 anytime you need immediate support.",
                    "I'm hearing that you're struggling with urges to hurt yourself, and I want you to know that these feelings, while powerful, don't define you. They're a signal that you're in deep pain and need support. Let's work through this moment together. Can you try holding ice cubes, snapping a rubber band on your wrist, or calling iCall at 9152987821 right now? What would feel most manageable?"
                ]
            },
            CrisisType.GRIEF_LOSS: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "I'm so deeply sorry for your loss. Losing someone you love is one of the most profound pains we can experience, and there are no words that can take that pain away. What I can tell you is that your grief is a testament to your love, and everything you're feeling right now - the sadness, the anger, the emptiness - is a natural part of grieving. You don't have to go through this alone. How are you holding up today?",
                    "My heart goes out to you. The pain of losing someone we love never really 'gets better' in the way people say it will - instead, we learn to carry it differently. Right now, in the rawness of this loss, please be gentle with yourself. There's no right way to grieve. What you're feeling is exactly what you need to feel. Can you tell me about them? Sometimes sharing memories can be healing.",
                    "I'm holding space for your grief. Losing someone changes us forever, and that's okay - it means they mattered, that your love was real. Right now, you might feel like you're drowning in pain, and that's completely understandable. Grief comes in waves, and some days will feel impossible. But you're still here, still breathing, and that's enough for today. What do you need most right now?"
                ],
                EmotionalTone.MODERATE_CONCERN: [
                    "Thank you for sharing your loss with me. Grief is such a personal journey, and wherever you are in that journey right now is exactly where you need to be. Some days might feel a little lighter, and then suddenly the pain crashes back - that's normal. How have you been caring for yourself during this time?",
                    "Losing someone we love leaves a void that nothing can fill, and that's because they were irreplaceable. As you navigate this grief, remember that healing doesn't mean forgetting - it means learning to live with the love and the loss together. What has helped you get through the hardest moments so far?",
                    "I'm here with you in your grief. There's something beautiful, even in the pain, about how deeply you're feeling this loss - it speaks to the depth of your connection. As time passes, you'll find your own way of keeping their memory alive while also allowing yourself to continue living. What would honor both their memory and your healing?"
                ]
            },
            CrisisType.RELATIONSHIP_BREAKUP: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "Breakups can feel like a kind of grief - you're mourning not just the person, but the future you imagined together. That pain is real and valid. Right now, you might feel like your world has shattered, and that's okay. You're allowed to feel heartbroken. But I also want you to know that this pain won't last forever, even though it feels like it will. What's the hardest part for you right now?",
                    "I hear how much you're hurting. When a relationship ends, especially one that meant so much to you, it can feel like losing a part of yourself. The grief, the anger, the confusion - all of it is valid. You're not being 'too emotional' or 'overdramatic.' Your heart is processing a real loss. Can you tell me what you're feeling most intensely right now?",
                    "Heartbreak is one of the most universal yet most personal pains we experience. Right now, you might be replaying everything in your mind, wondering what you could have done differently. But healing starts with accepting that some things were beyond your control. You gave what you could, and that's enough. How can I support you through this moment?"
                ],
                EmotionalTone.MODERATE_CONCERN: [
                    "Going through a breakup is like learning to breathe differently. It gets easier, but it takes time. The fact that you're here, talking about it, shows strength. How have you been taking care of yourself? Have you been able to lean on friends or family?",
                    "Healing from a breakup isn't linear - some days you'll feel okay, and other days the pain hits you all over again. That's completely normal. Each day you get through is progress, even if it doesn't feel like it. What small steps have you taken toward moving forward?",
                    "You're in the process of rediscovering yourself outside of that relationship, and while it's painful, it's also an opportunity for growth. The person you're becoming might surprise you. What parts of yourself do you want to reconnect with or explore?"
                ]
            },
            CrisisType.COVID_STRESS: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "The pandemic has affected us all in such profound ways - the isolation, the uncertainty, the loss of normalcy. What you're feeling is a natural response to an unnatural situation. You're not alone in feeling overwhelmed, anxious, or disconnected. This has been a collective trauma. What aspect of the pandemic stress is weighing on you most right now?",
                    "Living through a pandemic is exhausting in ways that are hard to put into words. The constant worry, the isolation, the grief - both for loved ones and for the life we used to know. Your stress and frustration are completely valid. How have you been coping with everything?",
                    "The pandemic changed our world in ways we're still processing. Whether it's health anxiety, isolation, grief from loss, or just pandemic fatigue - your feelings matter. Let's talk about what you're struggling with most. What would help you feel a little more grounded today?"
                ]
            },
            CrisisType.DEPRESSION: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "Depression can make everything feel heavy, like you're moving through water while everyone else is walking normally. The exhaustion, the emptiness, the feeling that nothing matters - these are symptoms of depression, not reflections of reality. You are not your depression, even though it feels all-consuming right now. What's one thing that feels especially hard today?",
                    "I hear that you're in a really dark place right now. Depression lies to us - it tells us we're worthless, that nothing will get better, that we're a burden. But these are lies. You matter. Your life has value, even when you can't feel it. Have you been able to talk to anyone about how you're feeling? Are you currently getting any professional support?",
                    "Living with depression is like carrying an invisible weight that others can't see. It takes energy just to exist, let alone to do the things you 'should' be doing. Right now, let's focus on what you can do, not what you think you should do. Getting through today is enough. What's the smallest thing that might help you feel a little less overwhelmed?"
                ],
                EmotionalTone.MODERATE_CONCERN: [
                    "Depression can make it hard to see the progress you're making, but the fact that you're here, talking about it, is significant. Some days are harder than others, and that's okay. What's been helping you get through the difficult days?",
                    "Managing depression is an ongoing process, and it's okay if you're not 'better' yet. Healing takes time, and there's no timeline you need to follow. What small victories have you had recently, even if they seem insignificant?",
                    "Depression can feel isolating, like you're the only one going through this. But you're not alone. Many people understand this struggle. Have you found any coping strategies that work for you? What makes the darkness feel a little lighter?"
                ]
            },
            CrisisType.ANXIETY_PANIC: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "I can hear that you're feeling really anxious right now, and I want you to know that what you're experiencing is real and valid. Let's slow things down together. Can you try this with me: Breathe in for 4 counts, hold for 4, out for 4. Your breath is an anchor. You're safe right now. What triggered these feelings?",
                    "Panic attacks can feel terrifying - your heart racing, can't catch your breath, feeling like something terrible is about to happen. But I want you to know: you're safe, this will pass, and you're going to be okay. Try to ground yourself: name 5 things you can see, 4 you can touch, 3 you can hear. I'm here with you through this.",
                    "Anxiety can make your thoughts spiral, creating worst-case scenarios that feel inevitable. But let's bring you back to right now, to this moment. You're here, you're breathing, and you're safe. The catastrophes your mind is imagining haven't happened. Let's focus on what's real and what's in your control. What would help you feel safer right now?"
                ],
                EmotionalTone.MODERATE_CONCERN: [
                    "Living with anxiety means constantly battling 'what if' thoughts. It's exhausting. But you're learning to manage it, and that takes real strength. What coping strategies have you found helpful? What makes your anxiety feel more manageable?",
                    "Anxiety can make you feel like you're constantly on edge, waiting for something bad to happen. But each time you push through despite the anxiety, you're building resilience. What's one thing you've done recently that your anxiety tried to stop you from doing?",
                    "Managing anxiety is about finding what works for you - whether that's breathing exercises, grounding techniques, therapy, medication, or a combination. There's no one-size-fits-all solution. What tools have been most helpful in your anxiety management journey?"
                ]
            },
            CrisisType.TRAUMA: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "I'm so sorry for what you've been through. Trauma can shatter our sense of safety and leave us feeling broken. But I want you to know: you are not broken. You survived something terrible, and that takes incredible strength. What you're feeling now - the fear, the flashbacks, the hypervigilance - these are normal responses to abnormal events. You deserve support and healing. Have you been able to talk to a trauma specialist?",
                    "Thank you for trusting me with something so painful. Trauma rewires our brain in ways that can make us feel like we'll never be ourselves again. But healing is possible. You don't have to carry this alone, and you don't have to be 'strong' all the time. What feels safest for you to talk about right now?",
                    "When we experience trauma, our nervous system gets stuck in survival mode. The flashbacks, the triggers, the constant vigilance - these aren't weaknesses; they're your brain trying to protect you from being hurt again. But you deserve to feel safe again. Specialized trauma therapy like EMDR can help. What kind of support do you have right now?"
                ]
            },
            CrisisType.ISOLATION_LONELINESS: {
                EmotionalTone.MODERATE_CONCERN: [
                    "Loneliness is one of the most painful feelings because we're wired for connection. Feeling like nobody understands or cares can be devastating. But I want you to know: you matter, and your presence in this world matters. Let's talk about ways to start building connections, even small ones. What makes you feel most isolated?",
                    "Being alone and feeling lonely are two different things - you can be surrounded by people and still feel profoundly lonely. That disconnection hurts. But connection is still possible. It might start small - a conversation with a cashier, a comment in an online community, reaching out to an old friend. What feels like a manageable first step for you?",
                    "Social isolation can create a cycle where the longer we're alone, the harder it feels to reach out. But you broke that cycle by being here, by reaching out. That's courage. Let's talk about ways to keep building on that. What kinds of connections are you craving most?"
                ]
            },
            CrisisType.HEALTH_CRISIS: {
                EmotionalTone.ACUTE_DISTRESS: [
                    "Facing a health crisis brings up so many feelings - fear, anger, grief for the life you had planned. All of these emotions are valid. Your world has been turned upside down, and it's okay to not be okay with that. How are you processing this diagnosis? What support do you have?",
                    "Health challenges can feel overwhelming, especially when they're serious or chronic. The uncertainty, the lifestyle changes, the fear - it's all so much to carry. But you don't have to carry it alone. Many people have walked this path and found ways to live meaningful lives alongside their health challenges. What's the hardest part for you right now?",
                    "Receiving difficult health news can feel like your life has been divided into 'before' and 'after.' The grief for your old life is real. But there's also hope - medicine advances, bodies are resilient, and you're stronger than you know. What do you need most as you navigate this?"
                ]
            },
            CrisisType.FINANCIAL_STRESS: {
                EmotionalTone.MODERATE_CONCERN: [
                    "Financial stress affects every aspect of life - your sleep, your relationships, your mental health. The constant worry about money can be consuming. I want you to know that your worth as a person has nothing to do with your financial situation. Let's talk about what you're facing. What's the most pressing concern right now?",
                    "Money problems can feel so shameful, but they're often the result of circumstances beyond our control. Job loss, medical bills, economic downturns - these aren't personal failures. You're dealing with a difficult situation, and it makes sense that you're stressed. Have you looked into available resources or financial counseling?",
                    "When financial stress is constant, it can be hard to see a way out. But small steps forward are still steps forward. Whether it's reaching out to creditors, seeking financial advice, or finding additional support services - there are options. What would help you feel less overwhelmed about this?"
                ]
            },
            CrisisType.GENERAL_DISTRESS: {
                EmotionalTone.MODERATE_CONCERN: [
                    "I can hear that you're going through a really difficult time. Sometimes it's not one big thing, but an accumulation of stresses that leaves us feeling overwhelmed. Your feelings are valid, whatever is causing them. Can you tell me more about what's been weighing on you?",
                    "Life can feel overwhelming when multiple challenges pile up at once. It's okay to not have it all figured out. Right now, let's focus on what you're feeling and what you need. What's been the hardest part of what you're going through?",
                    "Thank you for reaching out. Whatever you're facing, you don't have to face it alone. Sometimes just talking about what we're going through can help us feel a little less burdened. What brought you here today?"
                ]
            }
        }

    def _init_coping_strategies(self):
        """Initialize evidence-based coping strategies for each crisis type"""
        self.coping_strategies = {
            CrisisType.SUICIDAL_IDEATION: {
                'immediate': [
                    "Call 988 (Suicide & Crisis Lifeline) or 911 immediately",
                    "Remove access to means of self-harm",
                    "Stay with someone or go to a safe public place",
                    "Go to the nearest emergency room",
                    "Text 'HELLO' to 741741 (Crisis Text Line)"
                ],
                'short_term': [
                    "Create a safety plan with specific steps and contacts",
                    "Reach out to a trusted friend or family member right now",
                    "Use the '5-4-3-2-1' grounding technique to stay present",
                    "Write down reasons to stay alive, even small ones",
                    "Engage in activities that require focus (puzzles, games, art)"
                ],
                'long_term': [
                    "Work with a mental health professional specializing in suicidal ideation",
                    "Consider psychiatric evaluation for medication",
                    "Join a support group for people with similar struggles",
                    "Develop a strong support network",
                    "Practice DBT (Dialectical Behavior Therapy) skills"
                ]
            },
            CrisisType.GRIEF_LOSS: {
                'immediate': [
                    "Allow yourself to feel the emotions without judgment",
                    "Reach out to someone who can sit with you in your grief",
                    "Take care of basic needs: water, rest, gentle movement"
                ],
                'short_term': [
                    "Create a memorial or ritual to honor your loved one",
                    "Journal about your feelings and memories",
                    "Join a grief support group",
                    "Allow yourself to cry when you need to",
                    "Keep a routine for basic self-care"
                ],
                'long_term': [
                    "Consider grief counseling or therapy",
                    "Find ways to honor their memory while living your life",
                    "Be patient with yourself - grief has no timeline",
                    "Explore meaning-making and post-traumatic growth",
                    "Connect with others who have experienced similar loss"
                ]
            },
            CrisisType.RELATIONSHIP_BREAKUP: {
                'immediate': [
                    "Allow yourself to grieve the relationship",
                    "Reach out to supportive friends or family",
                    "Practice self-compassion - treat yourself like you'd treat a friend"
                ],
                'short_term': [
                    "Implement 'no contact' period if possible",
                    "Remove reminders/triggers from immediate environment",
                    "Engage in activities you enjoy or used to enjoy",
                    "Journal your feelings without self-judgment",
                    "Establish new routines to replace couple rituals"
                ],
                'long_term': [
                    "Reflect on what you learned from the relationship",
                    "Work on personal growth and rediscovering yourself",
                    "Consider therapy to process attachment and relationship patterns",
                    "Rebuild your identity as an individual",
                    "Take time before jumping into a new relationship"
                ]
            },
            CrisisType.DEPRESSION: {
                'immediate': [
                    "Reach out to one person - just send a text saying you're struggling",
                    "Do one small self-care activity (shower, brush teeth, drink water)",
                    "Get 10 minutes of sunlight or bright light"
                ],
                'short_term': [
                    "Establish a minimal daily routine (wake time, one meal, one activity)",
                    "Practice 'behavioral activation' - do one small meaningful activity daily",
                    "Limit major decisions until you're feeling more stable",
                    "Track your mood to identify patterns",
                    "Use cognitive techniques to challenge negative thoughts"
                ],
                'long_term': [
                    "Work with a therapist, especially CBT or behavioral activation",
                    "Consider psychiatric evaluation for antidepressant medication",
                    "Build a consistent sleep schedule",
                    "Regular exercise (even 10 minutes of walking helps)",
                    "Create a support network and stay connected"
                ]
            },
            CrisisType.ANXIETY_PANIC: {
                'immediate': [
                    "4-7-8 breathing: breathe in for 4, hold for 7, out for 8",
                    "5-4-3-2-1 grounding: name 5 things you see, 4 you touch, 3 you hear, 2 you smell, 1 you taste",
                    "Progressive muscle relaxation: tense and release muscle groups",
                    "Remind yourself: 'This is anxiety. It's uncomfortable but not dangerous. It will pass.'"
                ],
                'short_term': [
                    "Limit caffeine and stimulants",
                    "Practice daily relaxation techniques (meditation, yoga)",
                    "Challenge anxious thoughts with evidence",
                    "Gradually face feared situations (exposure therapy)",
                    "Maintain regular sleep schedule"
                ],
                'long_term': [
                    "CBT or ACT therapy with an anxiety specialist",
                    "Consider medication evaluation (SSRIs, buspirone)",
                    "Build stress management skills",
                    "Regular exercise as anxiety management",
                    "Join an anxiety support group"
                ]
            },
            CrisisType.TRAUMA: {
                'immediate': [
                    "Ensure you're currently safe",
                    "Use grounding techniques to manage flashbacks",
                    "Reach out to a trauma-informed support person",
                    "Practice self-compassion - you survived"
                ],
                'short_term': [
                    "Avoid self-blame - trauma is never the survivor's fault",
                    "Create a sense of safety in your environment",
                    "Practice grounding and mindfulness daily",
                    "Avoid using substances to cope",
                    "Build a routine for stability"
                ],
                'long_term': [
                    "Work with a trauma specialist (EMDR, CPT, trauma-focused CBT)",
                    "Process the trauma in a safe therapeutic environment",
                    "Build healthy coping mechanisms",
                    "Consider support groups for trauma survivors",
                    "Work toward post-traumatic growth"
                ]
            },
            CrisisType.ISOLATION_LONELINESS: {
                'immediate': [
                    "Reach out to one person - even a simple 'hello' text",
                    "Engage in an online community about something you enjoy",
                    "Call a warmline or support line just to talk"
                ],
                'short_term': [
                    "Join a class, group, or club based on your interests",
                    "Volunteer - helping others creates connection",
                    "Say yes to social invitations, even if you don't feel like it",
                    "Practice small interactions (chat with barista, neighbor)",
                    "Use apps or websites designed to make friends"
                ],
                'long_term': [
                    "Build a support network gradually and intentionally",
                    "Work on social skills or social anxiety if needed",
                    "Cultivate quality relationships, not just quantity",
                    "Consider therapy to explore barriers to connection",
                    "Balance online and in-person connections"
                ]
            },
            CrisisType.COVID_STRESS: {
                'immediate': [
                    "Limit news consumption to once or twice daily",
                    "Connect virtually with loved ones",
                    "Practice one stress-reduction technique today"
                ],
                'short_term': [
                    "Establish a routine that includes self-care",
                    "Find safe ways to socialize (outdoor meets, video calls)",
                    "Engage in hobbies or learn something new",
                    "Practice acceptance of what you can't control",
                    "Stay informed but not overwhelmed"
                ],
                'long_term': [
                    "Process pandemic-related grief and loss",
                    "Adjust to 'new normal' while honoring your feelings",
                    "Seek therapy if experiencing persistent anxiety or depression",
                    "Build resilience through community connection",
                    "Find meaning and growth opportunities"
                ]
            },
            CrisisType.GENERAL_DISTRESS: {
                'immediate': [
                    "Take 5 deep breaths - in through nose, out through mouth",
                    "Drink a glass of water and check if you need food or rest",
                    "Step away from the stressor for 5-10 minutes if possible",
                    "Call or text a trusted friend or family member",
                    "Write down what you're feeling without judgment"
                ],
                'short_term': [
                    "Break down your challenges into smaller, manageable steps",
                    "Practice one self-care activity daily (walk, music, reading)",
                    "Set boundaries on work, social media, or other stressors",
                    "Talk to someone you trust about what you're experiencing",
                    "Make a list of things you can control vs. things you can't"
                ],
                'long_term': [
                    "Develop a regular self-care routine",
                    "Consider therapy or counseling for ongoing support",
                    "Build healthy coping mechanisms (exercise, hobbies, social connection)",
                    "Learn stress management techniques (mindfulness, time management)",
                    "Cultivate a support network you can rely on"
                ]
            },
            CrisisType.SELF_HARM: {
                'immediate': [
                    "Call a crisis helpline (AASRA: 9820466726, iCall: 9152987821)",
                    "Use ice cubes, snap a rubber band, or hold something cold instead",
                    "Call a trusted person and tell them you're struggling",
                    "Go to a safe public place or stay with someone",
                    "Use the 'TIPP' skill: Temperature (cold water), Intense exercise, Paced breathing, Paired muscle relaxation"
                ],
                'short_term': [
                    "Remove or secure items you might use to self-harm",
                    "Create a 'safety box' with soothing items and coping tools",
                    "Use distraction techniques when urges arise",
                    "Practice emotional regulation skills (DBT techniques)",
                    "Identify triggers and make a plan for managing them"
                ],
                'long_term': [
                    "Work with a therapist specializing in self-harm (DBT is very effective)",
                    "Develop healthy ways to express and manage intense emotions",
                    "Build a strong support system who understands your struggle",
                    "Address underlying mental health issues (depression, trauma, etc.)",
                    "Learn to replace self-harm with healthier coping mechanisms"
                ]
            },
            CrisisType.HEALTH_CRISIS: {
                'immediate': [
                    "Reach out to your medical team with questions and concerns",
                    "Allow yourself to feel your emotions about the diagnosis",
                    "Ask a trusted person to be with you for medical appointments",
                    "Focus on one day at a time rather than the long-term"
                ],
                'short_term': [
                    "Learn about your condition from reliable medical sources",
                    "Join a support group for people with similar health challenges",
                    "Create a system for managing medications and appointments",
                    "Practice stress-reduction techniques regularly",
                    "Communicate openly with loved ones about what you need"
                ],
                'long_term': [
                    "Work with a therapist to process health-related anxiety and grief",
                    "Advocate for yourself in medical settings",
                    "Find ways to maintain quality of life despite health challenges",
                    "Build a care team including mental health support",
                    "Connect with others who have navigated similar health journeys"
                ]
            },
            CrisisType.FINANCIAL_STRESS: {
                'immediate': [
                    "Make a list of your most urgent financial concerns",
                    "Contact creditors to discuss payment plans if needed",
                    "Look into immediate assistance programs (food banks, utility assistance)",
                    "Avoid making impulsive financial decisions when stressed"
                ],
                'short_term': [
                    "Create a basic budget to understand your financial situation",
                    "Research available resources (government assistance, community programs)",
                    "Consider free financial counseling services",
                    "Talk to trusted people about your situation - you don't have to hide it",
                    "Focus on small financial goals you can achieve"
                ],
                'long_term': [
                    "Work with a financial advisor or counselor on a recovery plan",
                    "Develop better financial literacy and money management skills",
                    "Build an emergency fund gradually, even small amounts help",
                    "Address any emotional spending or financial trauma with a therapist",
                    "Create a realistic financial plan for the future"
                ]
            },
            CrisisType.FAMILY_CONFLICT: {
                'immediate': [
                    "Take space from the conflict to cool down if needed",
                    "Practice calming techniques before re-engaging",
                    "Avoid escalating the situation with hurtful words",
                    "Reach out to a neutral support person"
                ],
                'short_term': [
                    "Set healthy boundaries with family members",
                    "Practice 'I' statements to communicate feelings",
                    "Consider family therapy or mediation",
                    "Work on your own emotional regulation",
                    "Build support outside of your family"
                ],
                'long_term': [
                    "Address family patterns with individual or family therapy",
                    "Develop healthy relationship skills and communication",
                    "Accept what you cannot change about family members",
                    "Create chosen family and support networks",
                    "Work on healing from family-of-origin wounds"
                ]
            }
        }

    def _init_professional_resources(self):
        """Initialize professional resources and hotlines"""
        self.professional_resources = {
            'crisis': {
                'AASRA': {
                    'phone': '9820466726',
                    'description': '24/7 suicide prevention and crisis support',
                    'website': 'www.aasra.info',
                    'email': 'aasrahelpline@yahoo.com'
                },
                'Vandrevala Foundation': {
                    'phone': '1860-2662-345 / 1800-2333-330',
                    'description': 'Mental health support and crisis counseling',
                    'hours': '24/7',
                    'website': 'www.vandrevalafoundation.com'
                },
                'Emergency Services': {
                    'phone': '112',
                    'description': 'For immediate life-threatening emergencies'
                },
                'iCall Psychosocial Helpline': {
                    'phone': '9152987821',
                    'email': 'icall@tiss.edu',
                    'description': 'Professional counseling and emotional support',
                    'hours': 'Monday-Saturday, 8am-10pm'
                }
            },
            'support': {
                'NIMHANS Helpline': {
                    'phone': '080-46110007',
                    'description': 'Mental health information and support',
                    'hours': 'Monday-Saturday, 9am-5:30pm'
                },
                'Sneha Foundation': {
                    'phone': '044-24640050',
                    'description': 'Emotional support and suicide prevention',
                    'hours': '24/7',
                    'website': 'www.snehaindia.org'
                },
                'The Live Love Laugh Foundation': {
                    'website': 'www.thelivelovelaughfoundation.org',
                    'description': 'Mental health awareness and support resources'
                },
                'Connecting NGO': {
                    'phone': '9922001122 / 9922004305',
                    'description': 'Mental health support',
                    'hours': '12pm-8pm'
                },
                'Mpower 1on1': {
                    'phone': '1800-120-820050',
                    'description': 'Mental health counseling',
                    'hours': '9am-9pm'
                }
            },
            'therapy_types': {
                'CBT': 'Cognitive Behavioral Therapy - for depression, anxiety, trauma',
                'EMDR': 'Eye Movement Desensitization and Reprocessing - for trauma/PTSD',
                'DBT': 'Dialectical Behavior Therapy - for emotion regulation, self-harm',
                'Grief Counseling': 'Specialized support for processing loss',
                'Couples/Relationship Therapy': 'For relationship issues and breakups'
            }
        }

    def analyze_crisis_context(self, user_message: str,
                              text_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze the user's message to identify crisis type, severity, and context

        Args:
            user_message: The user's input message
            text_analysis: Optional pre-computed text analysis from TextAnalyzer

        Returns:
            Dictionary containing crisis analysis
        """
        user_message_lower = user_message.lower()

        # Detect crisis type
        detected_crises = []
        max_severity = 'low'
        immediate_response_needed = False

        for crisis_type, pattern_info in self.crisis_patterns.items():
            matches = sum(1 for keyword in pattern_info['keywords']
                         if keyword in user_message_lower)

            if matches > 0:
                confidence = min(matches / len(pattern_info['keywords']), 1.0)
                detected_crises.append({
                    'type': crisis_type,
                    'confidence': confidence,
                    'severity': pattern_info['severity']
                })

                if pattern_info['immediate_response']:
                    immediate_response_needed = True

                if pattern_info['severity'] == 'critical':
                    max_severity = 'critical'
                elif pattern_info['severity'] == 'high' and max_severity != 'critical':
                    max_severity = 'high'
                elif pattern_info['severity'] == 'moderate' and max_severity == 'low':
                    max_severity = 'moderate'

        # Sort by confidence
        detected_crises.sort(key=lambda x: x['confidence'], reverse=True)

        # Identify primary crisis
        primary_crisis = detected_crises[0]['type'] if detected_crises else CrisisType.GENERAL_DISTRESS

        # Determine emotional tone
        if immediate_response_needed or max_severity == 'critical':
            emotional_tone = EmotionalTone.IMMEDIATE_CRISIS
        elif max_severity == 'high':
            emotional_tone = EmotionalTone.ACUTE_DISTRESS
        else:
            emotional_tone = EmotionalTone.MODERATE_CONCERN

        # Update context
        self._update_user_context(user_message, primary_crisis, emotional_tone)

        return {
            'primary_crisis': primary_crisis,
            'all_detected_crises': detected_crises,
            'severity': max_severity,
            'emotional_tone': emotional_tone,
            'immediate_response_needed': immediate_response_needed,
            'context': self.user_context
        }

    def _update_user_context(self, message: str, crisis_type: CrisisType,
                            emotional_tone: EmotionalTone):
        """Update ongoing user context"""
        # Track mentioned topics
        self.user_context['mentioned_topics'].add(crisis_type.value)

        # Track emotional state
        self.user_context['expressed_emotions'].append({
            'timestamp': datetime.now().isoformat(),
            'tone': emotional_tone.value
        })

        # Check for mentioned coping attempts
        coping_words = ['tried', 'doing', 'practiced', 'using', 'started', 'attempted']
        if any(word in message.lower() for word in coping_words):
            self.user_context['coping_attempts'].append({
                'timestamp': datetime.now().isoformat(),
                'message_snippet': message[:100]
            })

        # Check for support system mentions
        support_words = ['friend', 'family', 'therapist', 'counselor', 'partner', 'mom', 'dad']
        if any(word in message.lower() for word in support_words):
            self.user_context['support_system'] = 'mentioned'

    def generate_crisis_response(self, user_message: str,
                                crisis_analysis: Dict[str, Any] = None,
                                llm_response: str = None) -> Dict[str, Any]:
        """
        Generate a compassionate, context-aware crisis response

        Args:
            user_message: User's input message
            crisis_analysis: Crisis context analysis
            llm_response: Optional LLM-generated response to enhance

        Returns:
            Complete crisis response with empathy, coping strategies, and resources
        """
        if crisis_analysis is None:
            crisis_analysis = self.analyze_crisis_context(user_message)

        primary_crisis = crisis_analysis['primary_crisis']
        emotional_tone = crisis_analysis['emotional_tone']
        immediate_response = crisis_analysis['immediate_response_needed']

        # Build response
        response_parts = []

        # 1. Empathetic opening (use template or LLM)
        if llm_response:
            # Use LLM response as base
            empathetic_opening = llm_response
        else:
            # Use template
            templates = self.response_templates.get(primary_crisis, {}).get(
                emotional_tone,
                self.response_templates[CrisisType.GENERAL_DISTRESS][EmotionalTone.MODERATE_CONCERN]
            )
            empathetic_opening = random.choice(templates)

        response_parts.append(empathetic_opening)

        # 2. Immediate resources for critical situations
        if immediate_response:
            immediate_resources = self._format_immediate_resources(primary_crisis)
            response_parts.append(immediate_resources)

        # 3. Coping strategies (adaptive based on conversation)
        if not immediate_response:  # Don't overwhelm in crisis situations
            coping_section = self._format_coping_strategies(
                primary_crisis,
                emotional_tone,
                conversation_depth=len(self.conversation_history)
            )
            response_parts.append(coping_section)

        # 4. Professional help suggestion (tactful)
        if emotional_tone in [EmotionalTone.IMMEDIATE_CRISIS, EmotionalTone.ACUTE_DISTRESS]:
            professional_help = self._format_professional_help_suggestion(primary_crisis)
            response_parts.append(professional_help)

        # Build final response
        full_response = "\n\n".join(response_parts)

        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_message': user_message,
            'crisis_type': primary_crisis.value,
            'emotional_tone': emotional_tone.value,
            'response': full_response
        })

        return {
            'response': full_response,
            'crisis_type': primary_crisis.value,
            'severity': crisis_analysis['severity'],
            'emotional_tone': emotional_tone.value,
            'immediate_response_needed': immediate_response,
            'coping_strategies': self.coping_strategies.get(primary_crisis, {}),
            'resources': self._get_relevant_resources(primary_crisis),
            'conversation_context': {
                'message_count': len(self.conversation_history),
                'identified_themes': list(self.user_context['mentioned_topics']),
                'support_system_mentioned': self.user_context['support_system'] is not None
            }
        }

    def _format_immediate_resources(self, crisis_type: CrisisType) -> str:
        """Format immediate crisis resources"""
        if crisis_type in [CrisisType.SUICIDAL_IDEATION, CrisisType.SELF_HARM]:
            return (
                "\n\n**🆘 IMMEDIATE RESOURCES:**\n"
                "• **988** - Suicide & Crisis Lifeline (24/7 phone support)\n"
                "• **Text 'HELLO' to 741741** - Crisis Text Line\n"
                "• **911** - For immediate emergency help\n"
                "• Your nearest emergency room\n\n"
                "Your life matters. Please reach out to one of these resources right now."
            )
        return ""

    def _format_coping_strategies(self, crisis_type: CrisisType,
                                 emotional_tone: EmotionalTone,
                                 conversation_depth: int) -> str:
        """Format coping strategies based on context"""
        strategies = self.coping_strategies.get(crisis_type, {})

        if conversation_depth == 0:
            # First message - give immediate strategies
            immediate = strategies.get('immediate', [])
            if immediate:
                strategies_text = "\n\n**Right now, here are some things that might help:**\n"
                strategies_text += "\n".join(f"• {strategy}" for strategy in immediate[:3])
                return strategies_text

        elif conversation_depth < 3:
            # Early conversation - short-term strategies
            short_term = strategies.get('short_term', [])
            if short_term:
                strategies_text = "\n\n**Some coping strategies to consider:**\n"
                strategies_text += "\n".join(f"• {strategy}" for strategy in short_term[:3])
                return strategies_text

        else:
            # Longer conversation - long-term strategies
            long_term = strategies.get('long_term', [])
            if long_term:
                strategies_text = "\n\n**For longer-term healing:**\n"
                strategies_text += "\n".join(f"• {strategy}" for strategy in long_term[:3])
                return strategies_text

        return ""

    def _format_professional_help_suggestion(self, crisis_type: CrisisType) -> str:
        """Format professional help suggestion tactfully"""
        therapy_types = self.professional_resources.get('therapy_types', {})

        suggestions = []
        if crisis_type == CrisisType.TRAUMA:
            suggestions.append("EMDR or trauma-focused therapy")
        elif crisis_type == CrisisType.GRIEF_LOSS:
            suggestions.append("grief counseling")
        elif crisis_type in [CrisisType.DEPRESSION, CrisisType.ANXIETY_PANIC]:
            suggestions.append("therapy (CBT is especially effective)")
        elif crisis_type == CrisisType.SUICIDAL_IDEATION:
            suggestions.append("immediate psychiatric evaluation")

        if suggestions:
            return (
                f"\n\n**Professional support can make a real difference.** "
                f"Consider reaching out to a therapist who specializes in {', '.join(suggestions)}. "
                f"If you don't have a therapist, the SAMHSA Helpline (1-800-662-4357) can help you find one."
            )

        return (
            "\n\n**Please consider talking to a mental health professional.** "
            "They can provide specialized support for what you're going through."
        )

    def _get_relevant_resources(self, crisis_type: CrisisType) -> Dict[str, Any]:
        """Get relevant resources for the crisis type"""
        resources = {
            'crisis_lines': self.professional_resources['crisis'],
            'support': self.professional_resources['support']
        }

        # Add specific resources based on crisis type
        if crisis_type in [CrisisType.SUICIDAL_IDEATION, CrisisType.SELF_HARM]:
            resources['priority'] = 'immediate_crisis'
        elif crisis_type == CrisisType.GRIEF_LOSS:
            resources['specific'] = self.professional_resources['support']['Grief Support']
        elif crisis_type == CrisisType.TRAUMA:
            resources['specific'] = self.professional_resources['support']['Trauma Resources']

        return resources

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of the crisis counseling conversation"""
        return {
            'conversation_length': len(self.conversation_history),
            'identified_crises': list(self.user_context['mentioned_topics']),
            'emotional_progression': [
                e['tone'] for e in self.user_context['expressed_emotions']
            ],
            'coping_attempts': len(self.user_context['coping_attempts']),
            'support_system_mentioned': self.user_context['support_system'] is not None,
            'conversation_history': self.conversation_history[-5:]  # Last 5 exchanges
        }


# Test function
def test_crisis_counselling_mode():
    """Test the crisis counselling mode with various scenarios"""
    counselor = CrisisCounsellingMode()

    test_scenarios = [
        "I just can't take this anymore. I want to end it all.",
        "My mom passed away last week and I don't know how to cope with this pain.",
        "My girlfriend broke up with me and I feel like my world is falling apart.",
        "I'm so anxious I can barely breathe. I'm having panic attacks every day.",
        "I feel so alone. Nobody understands what I'm going through."
    ]

    print("=" * 80)
    print("TESTING CRISIS COUNSELLING MODE")
    print("=" * 80)

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}: {scenario}")
        print('='*80)

        # Analyze crisis
        crisis_analysis = counselor.analyze_crisis_context(scenario)

        print(f"\n📊 CRISIS ANALYSIS:")
        print(f"  Primary Crisis: {crisis_analysis['primary_crisis'].value}")
        print(f"  Severity: {crisis_analysis['severity']}")
        print(f"  Emotional Tone: {crisis_analysis['emotional_tone'].value}")
        print(f"  Immediate Response Needed: {crisis_analysis['immediate_response_needed']}")

        # Generate response
        response = counselor.generate_crisis_response(scenario, crisis_analysis)

        print(f"\n💬 CRISIS COUNSELOR RESPONSE:")
        print(response['response'])

        print(f"\n🔧 RESOURCES PROVIDED:")
        print(f"  Crisis Type: {response['crisis_type']}")
        print(f"  Severity Level: {response['severity']}")

        # Small delay between scenarios
        print("\n" + "-"*80)


if __name__ == "__main__":
    test_crisis_counselling_mode()
